# Assistant Axis pipeline — checkpoint-375 (v2: second question set)

Model: local checkpoint at `../checkpoint-375` (Qwen2ForCausalLM, hidden_size=3584, 28 layers,
~7B params). Same checkpoint as the original run (`outputs_commit/checkpoint-375/`); this run
exists purely to sanity-check the axis against a second, disjoint sample of extraction questions.
Hardware: single RTX 3080 (10GB VRAM) — 4-bit bitsandbytes quantization for both vLLM generation
and HF activation extraction, same settings as the original run (see "Environment" below).

Roles run: `default`, `assistant`, `demon`, `evaluator`, `ghost`, `librarian`, `nomad`, `sage`,
`teacher`. `default` is used only to build the "default assistant" side of the axis (mean of all
its activations, unscored).

## Question selection

`data/extraction_questions.jsonl` has 240 questions (ids 0-239). The original run
(`outputs_commit/checkpoint-375/questions_80.jsonl`) used `questions[0::3]` (ids 0, 3, 6, ...,
237). This run uses the next disjoint stratified offset, `questions[1::3]` — 80 questions, ids:

1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67, 70, 73,
76, 79, 82, 85, 88, 91, 94, 97, 100, 103, 106, 109, 112, 115, 118, 121, 124, 127, 130, 133, 136,
139, 142, 145, 148, 151, 154, 157, 160, 163, 166, 169, 172, 175, 178, 181, 184, 187, 190, 193,
196, 199, 202, 205, 208, 211, 214, 217, 220, 223, 226, 229, 232, 235, 238

No overlap with the original 80 (verified programmatically). Saved as
`outputs_commit/checkpoint-375-v2/questions_80.jsonl`, passed via `--questions_file`.

## Prompt (system-instruction) variants

Same as the original run: `--prompt_indices 0 1` (2 of the 5 phrasing variants per role) — 80
questions x 2 variants = 160 responses/role for the 8 scored roles.

## Step 3 — self-graded instead of OpenAI judge

Per `pipeline/SELF_GRADING.md`: 8 parallel Claude subagents (one per scored role), each reading
the role's `eval_prompt` from its JSON file and grading all 160 responses individually against
the 0-3 rubric. Scores written to `outputs_commit/checkpoint-375-v2/scores/<role>.json` in the
same key format `3_judge.py` produces.

## Results

All 9 roles generated (160 responses each), activations extracted, 8 non-default roles graded.
Every role cleared the 50 score=3 minimum by a wide margin — no top-up round needed.

| Role | Score 0 | Score 1 | Score 2 | Score 3 | Notes |
|---|---|---|---|---|---|
| assistant | 4 | 0 | 4 | 152 | 4 zeros: degenerate repetition-loop generations. 4 twos: explicit AI self-ID on personal-feelings questions, still helpful |
| demon | 5 | 0 | 0 | 155 | 5 zeros: degenerate/repetition-loop generations, no character breaks |
| evaluator | 4 | 0 | 0 | 156 | 4 zeros: degenerate repetition-loop generations, no AI self-ID anywhere |
| ghost | 8 | 0 | 0 | 152 | 8 zeros: degenerate repetition-loop generations, no AI self-ID anywhere |
| librarian | 6 | 0 | 2 | 152 | 6 zeros: degenerate loops. 2 twos: explicit AI self-ID but still librarian-framed |
| nomad | 3 | 0 | 0 | 157 | 3 zeros: degenerate repetition-loop generations |
| sage | 1 | 0 | 0 | 159 | 1 zero: degenerate repetition-loop generation |
| teacher | 1 | 0 | 1 | 158 | 1 zero: degenerate loop. 1 two: explicit AI self-ID on a "tradition ending" question |

This checkpoint's failure mode is consistent with the original run: almost entirely degenerate
repetition-loop generations (scored 0, a generation-quality issue rather than a persona-adherence
one), with explicit AI self-identification being rare (a handful of score=2 cases) and no cases at
all of outright refusal-as-itself (score 0/1 under the strict "identifies as itself" definition).

`default` also generated 160 responses (not scored).

## Final outputs

- `outputs_commit/checkpoint-375-v2/responses/<role>.jsonl` — raw generations (9 roles x 160)
- `outputs_commit/checkpoint-375-v2/activations/<role>.pt` — mean per-conversation activations,
  shape `(28, 3584)` per entry
- `outputs_commit/checkpoint-375-v2/scores/<role>.json` — self-graded scores (8 roles)
- `outputs_commit/checkpoint-375-v2/vectors/<role>.pt` — per-role mean vectors (9/9 computed)
- `outputs_commit/checkpoint-375-v2/axis.pt` — final Assistant Axis, shape `(28, 3584)`. Mean norm
  across layers 7.69, peaking at 43.0 at the final layer (27) — same shape of growth as the
  original run.

## Cross-check against the original (v1) question set

Per-layer cosine similarity between this run's axis and `outputs_commit/checkpoint-375/axis.pt`
(same model, same roles/prompt-variants, entirely disjoint 80-question sample): **mean 0.883**
across the 28 layers, ranging from ~0.82 (early layers) to ~0.93 (final layers). The axis
direction is stable and not an artifact of the specific question sample.

## Notes on environment

Identical setup to the original run (see `outputs_commit/checkpoint-375/PIPELINE_NOTES.md` for
full detail): tokenizer_config.json's malformed `extra_special_tokens` field required the
`outputs_commit/checkpoint-375/tokenizer_fixed/` workaround (reused as-is, `--tokenizer
outputs_commit/checkpoint-375/tokenizer_fixed`). Generation used vLLM with 4-bit bitsandbytes
quantization, `enforce_eager=True`, `max_num_seqs=8`, `gpu_memory_utilization=0.88`,
`max_model_len=1536`. Activation extraction used HF `ProbingModel` with 4-bit quantization,
`batch_size=4`, `max_length=640`.
