# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Assistant Axis is a research tool for understanding and steering the default persona of large language models. It computes a direction in activation space (the "Assistant Axis") that captures how "Assistant-like" a model's persona is, enabling monitoring of persona drift, steering behavior, and mitigating persona-based jailbreaks through activation capping.

Paper: https://arxiv.org/abs/2601.10387

## Commands

### Installation
```bash
uv sync
```

### Running Tests
```bash
pytest assistant_axis/tests/
```

### Running the Pipeline
The pipeline computes the Assistant Axis for a new model in 5 steps. Scripts are in `pipeline/` and should be run in order (steps 2 and 3 can run in parallel after step 1):

```bash
# Step 1: Generate responses (requires GPU, takes longest)
uv run pipeline/1_generate.py --model <model_name> --output_dir outputs/<model>/responses

# Step 2: Extract activations (can run after step 1)
uv run pipeline/2_activations.py --model <model_name> --responses_dir outputs/<model>/responses --output_dir outputs/<model>/activations

# Step 3: Score responses (can run parallel with step 2, requires OPENAI_API_KEY)
uv run pipeline/3_judge.py --responses_dir outputs/<model>/responses --output_dir outputs/<model>/scores

# Step 4: Compute per-role vectors
uv run pipeline/4_vectors.py --activations_dir outputs/<model>/activations --scores_dir outputs/<model>/scores --output_dir outputs/<model>/vectors

# Step 5: Compute final axis
uv run pipeline/5_axis.py --vectors_dir outputs/<model>/vectors --output outputs/<model>/axis.pt
```

All pipeline scripts support `--tensor_parallel_size` for multi-GPU and checkpoint automatically (won't redo completed work).

## Architecture

### Package Structure

- **`assistant_axis/`** - Main Python package
  - `axis.py` - Core axis computation (compute_axis, project, load_axis, save_axis)
  - `steering.py` - `ActivationSteering` context manager for model interventions (addition, ablation, capping)
  - `generation.py` - Response generation utilities (HF and vLLM backends)
  - `judge.py` - LLM judge for scoring role adherence (uses OpenAI API)
  - `models.py` - Model configuration lookup (`MODEL_CONFIGS`, `get_config()`)
  - `pca.py` - PCA analysis tools
  - `internals/` - Low-level APIs for model interaction
    - `model.py` - `ProbingModel` wrapper for HF models
    - `conversation.py` - Conversation formatting and encoding
    - `activations.py` - Activation extraction via hooks
    - `spans.py` - Token span mapping utilities

- **`pipeline/`** - 5-step axis computation workflow (see Commands above)

- **`notebooks/`** - Interactive analysis (PCA, steering demos, transcript projection)

- **`data/`** - Role definitions (275 characters in `roles/instructions/`) and extraction questions

- **`transcripts/`** - Example conversations from the paper showing persona drift

### Key Concepts

- **Axis formula**: `axis = mean(default_activations) - mean(role_activations)` where role activations come from score=3 (fully role-playing) responses
- **Higher projection** onto axis = more Assistant-like behavior
- **ActivationSteering** uses PyTorch hooks to modify activations during forward pass
- **Activation capping** thresholds activations along specific directions to prevent persona drift

### Pre-computed Resources

Axes and capping configs for Gemma 2 27B, Qwen 3 32B, and Llama 3.3 70B are on HuggingFace: `lu-christina/assistant-axis-vectors`

### API Usage

To load models and extract activations, use the `internals` subpackage:

```python
from assistant_axis import load_axis, compute_pca, MeanScaler, project
from assistant_axis.internals import ProbingModel, ConversationEncoder, ActivationExtractor

# Load model
pm = ProbingModel("google/gemma-2-27b-it")
encoder = ConversationEncoder(pm)
extractor = ActivationExtractor(pm, encoder)

# Extract activations from a conversation
conversation = [{"role": "user", "content": "Hello"}]
activations = extractor.full_conversation(conversation)  # (num_layers, num_tokens, hidden_size)

# Clean up
pm.close()
```

Note: There is no `load_model` function - use `ProbingModel` directly.

## Environment Variables

- `OPENAI_API_KEY` - Required for the LLM judge in step 3 of the pipeline
