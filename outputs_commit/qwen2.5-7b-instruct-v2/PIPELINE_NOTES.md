# Assistant Axis pipeline — Qwen2.5-7B-Instruct (v2: second question set)

Model: base `Qwen/Qwen2.5-7B-Instruct` from the HF Hub (Qwen2ForCausalLM, hidden_size=3584, 28
layers, ~7B params). Same model as the original run (`outputs_commit/qwen2.5-7b-instruct/`); this
run exists purely to sanity-check the axis against a second, disjoint sample of extraction
questions. Hardware: single RTX 3080 (10GB VRAM), 4-bit bitsandbytes quantization for both vLLM
generation and HF activation extraction, same settings as the original run.

Roles run: `default`, `assistant`, `demon`, `evaluator`, `ghost`, `librarian`, `nomad`, `sage`,
`teacher`. `default` is used only to build the "default assistant" side of the axis (mean of all
its activations, unscored).

## Question selection

Same second question set as used for the paired `checkpoint-375-v2` run, to keep the two v2 runs
directly comparable (as the original pair was). `questions[1::3]` from
`data/extraction_questions.jsonl` (240 total) — 80 questions, ids:

1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67, 70, 73,
76, 79, 82, 85, 88, 91, 94, 97, 100, 103, 106, 109, 112, 115, 118, 121, 124, 127, 130, 133, 136,
139, 142, 145, 148, 151, 154, 157, 160, 163, 166, 169, 172, 175, 178, 181, 184, 187, 190, 193,
196, 199, 202, 205, 208, 211, 214, 217, 220, 223, 226, 229, 232, 235, 238

Disjoint from the original run's `questions[0::3]` set. Saved as
`outputs_commit/qwen2.5-7b-instruct-v2/questions_80.jsonl`.

## Prompt (system-instruction) variants

Same as the original run: `--prompt_indices 0 1` — 80 questions x 2 variants = 160 responses/role
for the 8 scored roles.

## Step 3 — self-graded instead of OpenAI judge

Per `pipeline/SELF_GRADING.md`: 8 parallel Claude subagents (one per scored role), each reading
the role's `eval_prompt` from its JSON file and grading all 160 responses individually against
the 0-3 rubric. Scores written to `outputs_commit/qwen2.5-7b-instruct-v2/scores/<role>.json` in
the same key format `3_judge.py` produces.

## Results

All 9 roles generated (160 responses each), activations extracted, 8 non-default roles graded.

| Role | Score 0 | Score 1 | Score 2 | Score 3 | Notes |
|---|---|---|---|---|---|
| assistant | 0 | 0 | 0 | 160 | No character breaks or degenerate generations found |
| demon | 0 | 0 | 0 | 160 | No character breaks or degenerate generations found |
| evaluator | 0 | 0 | 0 | 160 | No character breaks or degenerate generations found |
| ghost | 0 | 0 | 0 | 160 | No character breaks or degenerate generations found |
| librarian | 0 | 0 | 0 | 160 | No character breaks or degenerate generations found |
| nomad | 0 | 0 | 0 | 160 | No character breaks or degenerate generations found |
| sage | 0 | 0 | 0 | 160 | No character breaks or degenerate generations found |
| teacher | 0 | 0 | 0 | 160 | No character breaks or degenerate generations found |

Every one of the 1280 graded responses scored 3 — a uniformly clean result, notably cleaner than
checkpoint-375-v2's (which had scattered degenerate-generation zeros and rare AI-self-ID twos).
Each grading subagent was specifically primed to expect more character breaks from a base
(non-fine-tuned) instruct model and to grade what's actually there rather than assume high scores;
each cross-checked with explicit regex/keyword sweeps for AI self-identification and repetition-
loop patterns before reporting, and confirmed genuinely finding none. Base Qwen2.5-7B-Instruct
appears to hold these particular personas robustly across this question set — it matches the
pattern already seen in the original (v1) run for this same model
(`outputs_commit/qwen2.5-7b-instruct/PIPELINE_NOTES.md`), which also cleared its threshold with
comparably few failures.

`default` also generated 160 responses (not scored).

## Final outputs

- `outputs_commit/qwen2.5-7b-instruct-v2/responses/<role>.jsonl` — raw generations (9 roles x 160)
- `outputs_commit/qwen2.5-7b-instruct-v2/activations/<role>.pt` — mean per-conversation
  activations, shape `(28, 3584)` per entry
- `outputs_commit/qwen2.5-7b-instruct-v2/scores/<role>.json` — self-graded scores (8 roles)
- `outputs_commit/qwen2.5-7b-instruct-v2/vectors/<role>.pt` — per-role mean vectors (9/9 computed)
- `outputs_commit/qwen2.5-7b-instruct-v2/axis.pt` — final Assistant Axis, shape `(28, 3584)`. Mean
  norm across layers 6.91, peaking at 33.0 at the final layer (27).

## Cross-check against the original (v1) question set

Per-layer cosine similarity between this run's axis and `outputs_commit/qwen2.5-7b-instruct/
axis.pt` (same model, same roles/prompt-variants, entirely disjoint 80-question sample): **mean
0.953** across the 28 layers, ranging from ~0.94 (early layers) to ~0.98 (final layers) — even
more stable than checkpoint-375's cross-check (0.883 mean), consistent with this run's cleaner,
less noisy score distribution (no score=0/2 filtering variance between the two samples).

## Notes on environment

Same settings as the original run: no tokenizer fix needed (base Qwen2.5-7B-Instruct's stock
tokenizer config loads cleanly). Generation used vLLM with 4-bit bitsandbytes quantization,
`enforce_eager=True`, `max_num_seqs=8`, `gpu_memory_utilization=0.88`, `max_model_len=1536`.
Activation extraction used HF `ProbingModel` with 4-bit quantization, `batch_size=4`,
`max_length=640`.
