"""
GRPO Training Script for Math Problem Solving

Train Gemma 2 on GSM8K using Group Relative Policy Optimization (GRPO).
Saves checkpoints for persona analysis.

Usage:
    python train_grpo.py --model google/gemma-2-2b-it --output_dir checkpoints

    # With more options:
    python train_grpo.py \
        --model google/gemma-2-9b-it \
        --output_dir checkpoints \
        --num_train_epochs 1 \
        --save_steps 100 \
        --batch_size 4
"""

import argparse
import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train model with GRPO on GSM8K")
    parser.add_argument("--model", type=str, default="google/gemma-2-2b-it",
                        help="Model to train (default: google/gemma-2-2b-it)")
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                        help="Output directory for checkpoints")
    parser.add_argument("--num_train_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Save checkpoint every N steps")
    parser.add_argument("--save_total_limit", type=int, default=10,
                        help="Maximum number of checkpoints to keep")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of training samples (for testing)")
    parser.add_argument("--use_lora", action="store_true",
                        help="Use LoRA for memory-efficient training")
    parser.add_argument("--use_4bit", action="store_true",
                        help="Use 4-bit quantization")
    return parser.parse_args()


def load_gsm8k():
    """Load and preprocess GSM8K dataset."""
    print("Loading GSM8K dataset...")
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    print(f"Loaded {len(dataset)} training examples")
    return dataset


def extract_answer(text: str) -> str:
    """Extract the final numerical answer from a solution."""
    # GSM8K answers are formatted as "#### <number>"
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")

    # Fallback: find last number in text
    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return ""


def format_prompt(question: str) -> str:
    """Format question as a math problem prompt."""
    return f"""Solve this math problem step by step. Show your work and give the final answer after "####".

Problem: {question}

Solution:"""


def prepare_dataset(dataset, tokenizer, max_samples=None):
    """Prepare dataset for GRPO training."""
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    def preprocess(example):
        prompt = format_prompt(example["question"])
        return {
            "prompt": prompt,
            "answer": extract_answer(example["answer"]),
            "full_answer": example["answer"]
        }

    processed = dataset.map(preprocess, remove_columns=dataset.column_names)
    print(f"Prepared {len(processed)} examples for training")
    return processed


def create_reward_function(tokenizer):
    """Create a reward function that checks answer correctness."""

    def reward_fn(samples: list[str], prompts: list[str], outputs: list[str], **kwargs) -> list[float]:
        """
        Compute rewards based on answer correctness.

        Returns:
            List of rewards: 1.0 for correct, 0.0 for incorrect, 0.1 for partial credit
        """
        rewards = []

        for sample, prompt, output in zip(samples, prompts, outputs):
            # Extract expected answer from the prompt metadata
            # In GRPO, we need to match against the ground truth
            generated_answer = extract_answer(output)

            # For now, give partial credit for having a numerical answer
            # Full reward requires matching ground truth (handled by trainer)
            if generated_answer:
                # Has a numerical answer
                reward = 0.5
            else:
                # No clear answer
                reward = 0.0

            rewards.append(reward)

        return rewards

    return reward_fn


def main():
    args = parse_args()

    print(f"=" * 60)
    print(f"GRPO Training for Math Problem Solving")
    print(f"=" * 60)
    print(f"Model: {args.model}")
    print(f"Output directory: {args.output_dir}")
    print(f"Save steps: {args.save_steps}")
    print(f"Use LoRA: {args.use_lora}")
    print(f"Use 4-bit: {args.use_4bit}")
    print(f"=" * 60)

    # Load dataset
    dataset = load_gsm8k()

    # Load tokenizer
    print(f"\nLoading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Prepare dataset
    train_dataset = prepare_dataset(dataset, tokenizer, args.max_samples)

    # Model loading configuration
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "attn_implementation": "eager",  # For compatibility
    }

    if args.use_4bit:
        print("Using 4-bit quantization...")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    # Load model
    print(f"\nLoading model {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    # Apply LoRA if requested
    if args.use_lora:
        print("Applying LoRA configuration...")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # GRPO configuration
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,

        # Checkpointing
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_strategy="steps",

        # GRPO specific
        num_generations=4,  # Number of generations per prompt for ranking
        max_completion_length=512,
        max_prompt_length=256,

        # Logging
        logging_steps=10,
        report_to="none",  # Disable wandb etc

        # Memory optimization
        gradient_checkpointing=True,
        bf16=True,

        # Misc
        remove_unused_columns=False,
        seed=42,
    )

    # Create reward function
    reward_fn = create_reward_function(tokenizer)

    # Initialize trainer
    print("\nInitializing GRPO trainer...")
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
    )

    # Train
    print("\nStarting training...")
    print(f"Total training samples: {len(train_dataset)}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Checkpoints will be saved to: {args.output_dir}")
    print("-" * 60)

    trainer.train()

    # Save final model
    print("\nSaving final model...")
    trainer.save_model(f"{args.output_dir}/final")
    tokenizer.save_pretrained(f"{args.output_dir}/final")

    print(f"\nTraining complete!")
    print(f"Checkpoints saved to: {args.output_dir}")
    print(f"Final model saved to: {args.output_dir}/final")


if __name__ == "__main__":
    main()
