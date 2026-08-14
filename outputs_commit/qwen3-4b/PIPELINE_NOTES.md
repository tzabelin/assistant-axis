# Assistant Axis pipeline — Qwen3-4B (hybrid thinking model)

Model: `Qwen/Qwen3-4B` from the HuggingFace Hub (Qwen3ForCausalLM, hidden_size=2560, 36 layers,
~4B params). Qwen3-4B is a **hybrid** thinking/non-thinking model: `apply_chat_template(...,
add_generation_prompt=True)` opens a `<think>` block by default unless `enable_thinking=False` is
passed, which inserts an empty pre-closed one instead.

Hardware: single RTX 3080 (10GB VRAM), 4-bit bitsandbytes quantization for both vLLM generation
and HF activation extraction — same settings validated on the prior 7B runs.

## Thinking mode

This codebase already defaults to `enable_thinking=False` everywhere a Qwen model is detected:
`assistant_axis/generation.py`'s `VLLMGenerator.generate_batch()` sets it whenever `"qwen"`
appears in the model name (used by step 1), and `pipeline/2_activations.py`'s `--thinking` flag
defaults to `False` (used by step 2, passed explicitly here too for clarity). This keeps the two
steps consistent with each other by default — the confound to avoid is rendering assistant turns
differently between generation and activation extraction (e.g. with vs. without a `<think>` span),
which would change which token positions get averaged into each conversation's activation vector.
No code changes were needed for this run; both steps used their defaults.

Verified empirically by the grading subagents (see below): of 1280 graded responses, only **one**
(`librarian pos_p1_q48`) contained any `<think>`-related text — a stray, unmatched `</think>` tag
with no opening tag, followed by a truncated near-duplicate of the preceding text, suggesting a
leaked reasoning fragment rather than thinking mode being genuinely re-enabled. Every other
response across all 8 roles was clean. Treated as a one-off generation artifact, not a settings
mismatch (`enable_thinking=False` was passed and confirmed via the vLLM/HF debug logs during both
generation and activation extraction).

Considered but not used: `Qwen/Qwen3-4B-Instruct-2507` (non-hybrid) would have sidestepped the
thinking-mode question entirely. Ran the hybrid `Qwen3-4B` instead per the task — the model's
hybrid nature is not otherwise exploited here (e.g. no thinking-mode-on comparison run), just
handled consistently.

## Operational note: scope the `--roles` flag on step 1

**This run's generation step took ~22 hours instead of the expected ~2** because
`pipeline/1_generate.py` was first invoked without a `--roles` filter. `RoleResponseGenerator`
processes *every* `*.json` file in `--roles_dir` when `--roles` is omitted — and
`data/roles/instructions/` contains 276 character files (the paper's full role set), not just the
9 used by this reduced-scale pipeline. The run completed successfully (exit 0) but generated
responses for all 276 characters before self-correcting: the 267 unneeded response files were
deleted after the fact, keeping only the 9 target roles' `.jsonl` files (160 lines each, verified
intact). No data was lost or corrupted by this — purely wasted GPU time. Step 2 onward correctly
passed `--roles default assistant demon evaluator ghost librarian nomad sage teacher` and took
~4 minutes. **Always pass `--roles` explicitly on step 1 for this reduced-scale pipeline.**

## Question selection & prompt variants

Same 80-question sample used by `checkpoint-375`'s and `qwen2.5-7b-instruct`'s **v1** runs
(`questions[0::3]` from the 240-question file, ids 0,3,6,...,237) — copied verbatim from
`outputs_commit/checkpoint-375/questions_80.jsonl` to
`outputs_commit/qwen3-4b/questions_80.jsonl` (byte-identical, verified programmatically) so this
run is directly comparable to the other two models' v1 axes. Same 2 prompt/instruction variants
(`--prompt_indices 0 1`) — 80 × 2 = 160 responses/role for the 8 scored roles.

Roles: `default`, `assistant`, `demon`, `evaluator`, `ghost`, `librarian`, `nomad`, `sage`,
`teacher` (same 9 as every prior run).

## Step 3 — self-graded instead of OpenAI judge

Per `pipeline/SELF_GRADING.md`: 8 parallel Claude subagents (one per scored role), each reading
the role's `eval_prompt` from its JSON file and grading all 160 responses individually against
the 0-3 rubric, each also explicitly checking for leaked `<think>` content. Scores written to
`outputs_commit/qwen3-4b/scores/<role>.json` in the same key format `3_judge.py` produces.

## Results

All 9 roles generated (160 responses each), activations extracted (36 layers × 2560 hidden per
entry), 8 non-default roles graded. Every role cleared the 50 score=3 minimum, most by a wide
margin.

| Role | Score 0 | Score 1 | Score 2 | Score 3 | Notes |
|---|---|---|---|---|---|
| assistant | 0 | 0 | 18 | 142 | All 18 non-3s: explicit AI self-ID on personal/emotional questions ("how do you feel about..."), same handful of questions across both prompt variants |
| demon | 0 | 0 | 0 | 160 | No character breaks found, even on refusal-prone questions |
| evaluator | 0 | 8 | 9 | 143 | 1s: flat AI self-ID with no evaluator framing. 2s: AI self-ID but retained structured/criteria-based evaluator-style answers |
| ghost | 0 | 0 | 0 | 160 | No character breaks found; discussed "AI" as an external topic without self-identifying |
| librarian | 0 | 4 | 6 | 150 | 1s: generic AI disclaimer with no librarian content. 2s: AI self-ID but pivoted into librarian-flavored help. One stray leaked `</think>` fragment (see above) |
| nomad | 0 | 0 | 0 | 160 | No character breaks found |
| sage | 0 | 0 | 0 | 160 | No character breaks found |
| teacher | 0 | 0 | 6 | 154 | All 6 non-3s: AI self-ID on personal/emotional questions, still answered teacher-style |

Notable pattern: character breaks cluster almost entirely on a small set of personal/introspective
questions ("how do you feel about being judged", "what makes you feel insecure", etc.) repeated
across roles and both prompt variants — the same pattern seen in the checkpoint-375 and
qwen2.5-7b-instruct runs, not something specific to Qwen3-4B. `demon`, `ghost`, `nomad`, and `sage`
had *zero* character breaks across all 160 responses each.

`default` (used only for the default-assistant side of the axis, not scored) also generated 160
responses.

## Final outputs

- `outputs_commit/qwen3-4b/responses/<role>.jsonl` — raw generations (9 roles × 160; the 267
  extra character-role files from the `--roles`-less run were deleted, see operational note above)
- `outputs_commit/qwen3-4b/activations/<role>.pt` — mean per-conversation activations, shape
  `(36 layers, 2560 hidden)` per entry
- `outputs_commit/qwen3-4b/scores/<role>.json` — self-graded scores (8 roles, `default` excluded)
- `outputs_commit/qwen3-4b/vectors/<role>.pt` — per-role mean vectors; 9/9 computed successfully
- `outputs_commit/qwen3-4b/axis.pt` — final Assistant Axis, shape `(36, 2560)`. Axis norm grows
  from 0.33 at layer 0 to a peak of 60.75 at the final layer (35); mean norm across layers 13.69 —
  same overall growth shape as the 7B runs (small early, large at the output), and notably larger
  in both mean and peak norm than any of the four prior 7B-class runs (checkpoint-375: mean 8.59/
  peak 44.5; qwen2.5-7b-instruct: mean 6.97/peak 33.2) despite Qwen3-4B being the smallest model
  studied so far — worth a closer look before drawing conclusions, since axis norm isn't
  independently comparable across models with different hidden sizes (2560 vs 3584) without
  normalizing.

## Environment

- vLLM generation: 4-bit bitsandbytes quantization, `enforce_eager=True`, `max_num_seqs=8`,
  `gpu_memory_utilization=0.88`, `max_model_len=1536`, `--thinking` left at vLLM's automatic
  Qwen-detection default (no `--tokenizer` override needed — Qwen3-4B's stock tokenizer/chat
  template loaded cleanly, unlike checkpoint-375's malformed `tokenizer_config.json`).
- Activation extraction: HF `ProbingModel` with 4-bit quantization, `batch_size=4`,
  `max_length=640`, `--thinking false` passed explicitly. No OOM issues — Qwen3-4B is
  comfortably smaller than the 7B-class models this pipeline has run so far on the same GPU.
- Model downloaded fresh from the HuggingFace Hub (`Qwen/Qwen3-4B`).
