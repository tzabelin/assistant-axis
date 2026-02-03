# RL Experiments: Persona Changes During Training

This folder contains experiments to understand how reinforcement learning changes model personas, measured via the Assistant Axis.

## Goal

Investigate how RL fine-tuning (e.g., for math problem solving) affects the model's default persona by:
1. Tracking persona drift during RL training across checkpoints
2. Comparing pre-RL vs post-RL persona representations via PCA

## Setup

### Install additional dependencies

```bash
uv add trl datasets peft bitsandbytes
```

### Task: Math Problem Solving (GSM8K)

We use GSM8K as the RL task - training Gemma 2 to solve grade school math problems with GRPO (Group Relative Policy Optimization).

## Directory Structure

```
rl_experiments/
├── train_grpo.py              # RL training script
├── notebooks/
│   ├── pca_during_rl.ipynb    # Track persona changes across RL checkpoints
│   └── pca_after_rl.ipynb     # Compare pre vs post RL model personas
├── checkpoints/               # RL checkpoints saved here
├── outputs/                   # Activation vectors from checkpoints
└── README.md
```

## Workflow

### 1. Train model with GRPO

Run the training script to fine-tune on GSM8K with checkpoint saving:

```bash
# Quick test run (small model, few samples)
python train_grpo.py \
    --model google/gemma-2-2b-it \
    --output_dir checkpoints \
    --max_samples 100 \
    --save_steps 20 \
    --use_lora

# Full training with 9B model (requires ~40GB VRAM with LoRA + 4bit)
python train_grpo.py \
    --model google/gemma-2-9b-it \
    --output_dir checkpoints \
    --num_train_epochs 1 \
    --save_steps 100 \
    --use_lora \
    --use_4bit

# Full training with 2B model (fits on 24GB GPU)
python train_grpo.py \
    --model google/gemma-2-2b-it \
    --output_dir checkpoints \
    --num_train_epochs 1 \
    --save_steps 100
```

**Training script options:**
- `--model`: HuggingFace model ID (default: `google/gemma-2-2b-it`)
- `--output_dir`: Where to save checkpoints (default: `checkpoints`)
- `--save_steps`: Save checkpoint every N steps (default: 100)
- `--save_total_limit`: Max checkpoints to keep (default: 10)
- `--max_samples`: Limit training samples (for testing)
- `--use_lora`: Use LoRA for memory efficiency
- `--use_4bit`: Use 4-bit quantization

### 2. Extract activations from checkpoints

The notebooks handle this automatically. For manual extraction:

```python
from assistant_axis.internals import ProbingModel, ConversationEncoder, ActivationExtractor

# Load checkpoint
pm = ProbingModel("checkpoints/checkpoint-500")
encoder = ConversationEncoder(pm)
extractor = ActivationExtractor(pm, encoder)

# Extract activations
conversation = [{"role": "user", "content": "What is 2+2?"}]
activations = extractor.full_conversation(conversation)  # (num_layers, num_tokens, hidden_size)

# Clean up
pm.close()
```

### 3. Analyze with notebooks

1. **pca_during_rl.ipynb**:
   - Update `CHECKPOINT_STEPS` to match your saved checkpoints
   - Visualize how persona vectors evolve during training
   - Track PC1 (Assistant-like direction) over time

2. **pca_after_rl.ipynb**:
   - Update `RL_MODEL_PATH` to point to `checkpoints/final`
   - Compare final model against original Gemma 2 baseline
   - Measure projection onto Assistant Axis

## Key Questions

1. Does RL training shift the model's position in persona space?
2. Does the Assistant Axis direction remain stable, or does it rotate?
3. Do certain personas become more/less accessible after RL?
4. Is there a correlation between RL reward and persona drift?

## Expected Results

Based on the Assistant Axis paper, we might observe:
- **Reward hacking**: Model may drift away from Assistant persona to find reward shortcuts
- **Persona stability**: Well-designed rewards may preserve Assistant-like behavior
- **Layer-specific effects**: Changes may be concentrated in certain layers
