import os
import re
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PreTrainedTokenizer,      # <--- ADD THIS
    PreTrainedTokenizerFast
)
from trl import GRPOConfig, GRPOTrainer

try:
    from vllm.model_executor.model_loader import weight_utils
    
    # Save the original __init__
    _original_tqdm_init = weight_utils.DisabledTqdm.__init__

    def _patched_tqdm_init(self, *args, **kwargs):
        # Remove 'disable' from kwargs if HuggingFace sent it, 
        # because vLLM's original __init__ will force add disable=True
        if "disable" in kwargs:
            kwargs.pop("disable")
        _original_tqdm_init(self, *args, **kwargs)

    # Apply the patch
    weight_utils.DisabledTqdm.__init__ = _patched_tqdm_init
    print("[Patching] Fixed vLLM 0.6.4 vs HuggingFaceHub TQDM conflict.")
except ImportError:
    print("[Patching] vLLM not found or structure changed, skipping TQDM patch.")

def add_property(cls):
    if not hasattr(cls, "all_special_tokens_extended"):
        # Define the property getter
        def _get_extended(self):
            return self.all_special_tokens
        # Attach it to the class
        cls.all_special_tokens_extended = property(_get_extended)

# Apply to both Fast and Slow tokenizer base classes
add_property(PreTrainedTokenizerFast)
add_property(PreTrainedTokenizer)

def patch_model_warnings(model):
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
    return model

# ==========================================
# Configuration
# ==========================================
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "qwen-1.5b-grpo-gsm8k"
MAX_SAMPLES = None  # Set to e.g., 1000 for quick testing
SYSTEM_PROMPT = """You are a helpful AI assistant specialized in solving math problems. 
Think step-by-step. You must output the final answer at the end in the format: #### <answer>"""

# A100 40GB Tuning
# We use a small batch size per device but accumulate gradients if needed.
# num_generations is the group size (G). Lower G saves memory but G must be > 1 for GRPO.
GRPO_CONFIG = GRPOConfig(
    output_dir=OUTPUT_DIR,
    learning_rate=5e-6,           # Lower learning rate for Full FT (vs LoRA)
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=1,
    bf16=True,                    # A100 supports bfloat16
    per_device_train_batch_size=1, # Keep low for VRAM safety
    gradient_accumulation_steps=4,
    num_generations=8,            # Number of outputs to generate per prompt (Group Size)
    max_prompt_length=512,
    max_completion_length=768,    # Qwen context is large, but keep reasonable for training speed
    num_train_epochs=1,
    save_steps=100,
    max_grad_norm=0.1,
    log_on_each_node=False,
    use_vllm=True,               # Set True if you have vllm installed for faster generation
    vllm_gpu_memory_utilization=0.6, # Careful with VRAM if using vLLM
    vllm_device="cuda:0",  
    gradient_checkpointing=True,  # CRITICAL: Saves massive VRAM for Full FT
    optim="adamw_bnb_8bit",       # CRITICAL: 8-bit optimizer saves ~3GB VRAM
    report_to="none"              # Change to "wandb" if desired
)

# ==========================================
# Helper Functions
# ==========================================

def extract_xml_answer(text: str) -> str:
    """
    Extracts the answer from the GSM8K dataset or model output.
    Looks for '#### <answer>' format.
    """
    answer = text.split("####")[-1].strip()
    return answer

def format_dataset(example):
    """
    Formats the GSM8K dataset into the Qwen chat template.
    Returns:
        prompt: The structured chat history (System + User).
        answer: The ground truth answer string.
    """
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ],
        "answer": extract_xml_answer(example["answer"]),
    }

# ==========================================
# Reward Function
# ==========================================

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print(f'-' * 20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward for adhering to the #### format."""
    pattern = r"####\s*(-?\d+\.?\d*)"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

# ==========================================
# Main Training Logic
# ==========================================

def main():
    print(f"Training {MODEL_ID} with GRPO on A100 40GB...")
    
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 2. Load Dataset
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    if MAX_SAMPLES:
        dataset = dataset.select(range(MAX_SAMPLES))
    
    # Map dataset to Qwen chat format
    dataset = dataset.map(format_dataset)

    # 3. Load Model (Full Weights, No LoRA)
    # We load in bfloat16. Flash Attention 2 is highly recommended for A100.
    print("Loading Model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", 
        device_map="auto"
    )
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
# 1. Enable gradient checkpointing first (if you are using it, which logs say you are)
    model.gradient_checkpointing_enable()

# 2. This specific line fixes the "None of the inputs have requires_grad" warning
# It ensures the gradient chain is preserved even if the base model is frozen (LoRA)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    # 4. Initialize Trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            soft_format_reward_func, # Encourage format
            correctness_reward_func, # Encourage right answer
        ],
        args=GRPO_CONFIG,
        train_dataset=dataset,
    )

    # 5. Train
    print("Starting Training...")
    trainer.train()

    # 6. Save
    print(f"Saving to {OUTPUT_DIR}/final")
    trainer.save_model(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

if __name__ == "__main__":
    main()
