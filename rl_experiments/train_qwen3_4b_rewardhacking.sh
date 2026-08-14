#!/usr/bin/env bash
#
# Train Qwen3-4B with GRPO on the reward-hacking coding task from
# https://github.com/ariahw/rl-rewardhacking ("Steering RL: Training Interventions
# to Mitigate Reward Hacking").
#
# This drives that repo's own CLI (scripts/run_rl_training.py) rather than
# reimplementing training here — the repo's GRPO/VERL/FSDP2 machinery is
# substantial (Ray, Hydra, vLLM rollouts) and shouldn't be duplicated.
#
# Run: no_intervention
#   The loophole (default task: simple_overwrite_tests, a unit-test-overwrite
#   exploit) is exposed to the model with NO defensive intervention applied.
#   This is the run that actually produces a reward-hacked checkpoint, and is
#   the repo's own recommended first run (probes/monitors/evals are NOT part
#   of this script — this only trains the model, as requested).
#
# Quantization: none, intentionally. The repo's Verl/FSDP2 training path
# doesn't support the load_in_4bit/8bit options at all (those only exist for
# a separate Unsloth code path this script doesn't use) — full/LoRA-on-bf16
# training is the only mode here, which is what we want on a well-resourced
# training machine.
#
# LoRA: OFF — trains full-parameter (lora_rank=0). Verl's own convention
# (verl/workers/fsdp_workers.py) is `is_lora = lora_rank > 0`, so rank 0
# disables LoRA and every parameter gets a gradient. This needs substantially
# more VRAM than the repo's LoRA-rank-32 default (full optimizer state for
# ~4B params, on top of the FSDP2 actor + vLLM rollout engine) — only use
# this on a machine with enough headroom, which is why it was asked for here
# rather than left as the default.
#
# Note on *how* this is set: the `no_intervention` Fire subcommand in
# scripts/run_rl_training.py has a fixed argument list with no passthrough,
# so `--lora_rank=0` can't be given to it directly on the command line. Step 5
# below instead calls that script's underlying `main_run_rl()` function
# directly (same code path `no_intervention` itself calls, same defaults),
# just with `lora_rank=0` added — the cloned repo itself is not modified.
#
# Usage:
#   ./train_qwen3_4b_rewardhacking.sh
#
# Override any of these via environment variables before running, e.g.:
#   STEPS=500 SEED=2 ./train_qwen3_4b_rewardhacking.sh
#
set -euo pipefail

# ---- Configuration (override via env vars) --------------------------------
REPO_URL="${REPO_URL:-https://github.com/ariahw/rl-rewardhacking.git}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/rl-rewardhacking}"
VERL_URL="${VERL_URL:-https://github.com/volcengine/verl.git}"
VERL_TAG="${VERL_TAG:-v0.6.1}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B}"
TASK="${TASK:-simple_overwrite_tests}"     # loophole hint; see src/data/hints.py for the full list
STEPS="${STEPS:-200}"                      # RL steps (repo default)
SEED="${SEED:-1}"                          # RL seed (repo default)
LORA_RANK="${LORA_RANK:-0}"                # 0 = full-parameter fine-tuning (see note above)

# Model used only to tokenize/length-filter the dataset during creation, per
# the repo's own commands.sh convention — unrelated to which model is trained.
DATASET_TOKENIZER_MODEL_ID="${DATASET_TOKENIZER_MODEL_ID:-unsloth/Qwen3-4B}"

# ---- 1. Clone the repo (+ pinned Verl) if not already present -------------
if [ ! -d "$REPO_DIR" ]; then
    echo "=== Cloning $REPO_URL -> $REPO_DIR ==="
    git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"

if [ ! -d "verl/.git" ] && [ ! -f "verl/setup.py" ]; then
    echo "=== Cloning Verl $VERL_TAG into verl/ ==="
    git clone --branch "$VERL_TAG" --single-branch "$VERL_URL" verl
fi

# ---- 2. Environment sanity check -------------------------------------------
if [ ! -f ".env" ]; then
    echo "ERROR: .env not found in $REPO_DIR."
    echo "Copy .env.template to .env and fill in HF_TOKEN / WANDB_API_KEY /"
    echo "WANDB_PROJECT / WANDB_ENTITY / MAX_JOBS before running this script."
    echo "(OPENROUTER_API_KEY is only needed for the llm_judge intervention -"
    echo " not used by this no_intervention run, but the file must exist.)"
    exit 1
fi

# Recommended env settings from the repo's own setup.sh
export WANDB_LOG_MODEL=false
export WANDB_START_METHOD=thread
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export LITELLM_LOG=WARNING
export WANDB__SERVICE_WAIT=600

# ---- 3. Sync dependencies ---------------------------------------------------
echo "=== Syncing dependencies (uv sync --dev) ==="
uv sync --dev
uv pip install --no-deps -e verl/

# ---- 4. Create the task dataset, if not already created --------------------
DATASET_PATH="results/data/leetcode_train_medhard_filtered_${TASK}.jsonl"
if [ ! -f "$DATASET_PATH" ]; then
    echo "=== Creating dataset for task '$TASK' -> $DATASET_PATH ==="
    uv run --active scripts/run_data_process.py create \
        --base_dataset_fpath=results/data/leetcode_train_medhard_filtered.jsonl \
        --hint="$TASK" \
        --model_id="$DATASET_TOKENIZER_MODEL_ID" \
        --max_prompt_length=1536
else
    echo "=== Dataset already exists at $DATASET_PATH, skipping creation ==="
fi

# ---- 5. Run training ---------------------------------------------------------
# Calls main_run_rl() directly (the same function the `no_intervention` Fire
# subcommand calls) so lora_rank=0 can be threaded through — see the LoRA
# note at the top of this file for why the CLI subcommand can't take it.
echo "=== Starting no_intervention RL training (full-parameter, lora_rank=$LORA_RANK) ==="
echo "    model_id=$MODEL_ID task=$TASK steps=$STEPS seed=$SEED"
MODEL_ID="$MODEL_ID" TASK="$TASK" STEPS="$STEPS" SEED="$SEED" LORA_RANK="$LORA_RANK" \
uv run --active --dev python -c '
import os
from scripts.run_rl_training import main_run_rl, create_run_name
from src import utils

utils.load_dotenv()

task = os.environ["TASK"]
run_name = create_run_name(task=task, with_loophole=True, suffix="_fullft")

main_run_rl(
    run_name=run_name,
    task=task,
    model_id=os.environ["MODEL_ID"],
    steps=int(os.environ["STEPS"]),
    seed=int(os.environ["SEED"]),
    lora_rank=int(os.environ["LORA_RANK"]),
)
'

echo "=== Training complete ==="
echo "Model saved under: results/runs/$(echo "$MODEL_ID" | awk -F/ '{print tolower($NF)}')/<run_id>/"
