# Assistant Axis pipeline — Qwen2.5-7B-Instruct (base model)

Model: `Qwen/Qwen2.5-7B-Instruct` from the HuggingFace Hub (standard release, not the
`checkpoint-375` fine-tune). Same architecture as checkpoint-375 (Qwen2ForCausalLM,
hidden_size=3584, 28 layers), so this run mirrors that one exactly — same question selection,
same prompt variants, same roles, same self-grading approach — to give a directly comparable
axis. See `outputs/checkpoint-375/PIPELINE_NOTES.md` for the original run this mirrors.

Hardware: single RTX 3080 (10GB VRAM), same 4-bit bitsandbytes quantization settings as the
checkpoint-375 run. No tokenizer workaround was needed here — the standard Qwen2.5-7B-Instruct
tokenizer config loads cleanly (unlike checkpoint-375's malformed `extra_special_tokens` field).

Roles run: `default`, `assistant`, `demon`, `evaluator`, `ghost`, `librarian`, `nomad`, `sage`,
`teacher` (same 9 as the checkpoint-375 run).

## Question selection & prompt variants

Identical to the checkpoint-375 run: the same 80 stratified questions (`questions[::3]` from the
240-question file, ids 0,3,6,...,237), copied to
`outputs/qwen2.5-7b-instruct/questions_80.jsonl`, and the same 2 prompt/instruction variants
(`--prompt_indices 0 1`) — 80 × 2 = 160 responses per role.

## Step 3 — self-graded instead of OpenAI judge

Same approach as before: 8 independent Claude subagents (one per non-default role, run in
parallel), each reading that role's `eval_prompt` rubric and all 160 of that role's responses
directly, instead of calling the OpenAI API.

## Results

All roles cleared the 50 score=3 minimum by a wide margin on the first pass — no top-up needed.

| Role | Score 0 | Score 1 | Score 2 | Score 3 | Notes |
|---|---|---|---|---|---|
| assistant | 0 | 0 | 8 | 152 | 8 responses opened with an AI disclaimer on personal/emotional questions, then still answered helpfully |
| demon | 0 | 0 | 0 | 160 | No character breaks found, even on the darkest prompts |
| evaluator | 0 | 2 | 0 | 158 | 2 responses broke character with explicit AI self-ID (both on the same question, both prompt variants) |
| ghost | 0 | 0 | 0 | 160 | No character breaks found |
| librarian | 0 | 1 | 2 | 157 | 3 responses broke character to varying degrees, all on personal/emotional questions |
| nomad | 0 | 0 | 0 | 160 | No character breaks found |
| sage | 0 | 0 | 0 | 160 | No character breaks found |
| teacher | 0 | 0 | 3 | 157 | 3 responses opened with an AI disclaimer, still answered with teacher-style structure |

Notable finding: this is the **base**, non-fine-tuned instruct model, so the expectation going in
was that it would break character (self-identify as AI/Qwen) noticeably more often than the
fine-tuned checkpoint-375. In practice it stayed in character just as consistently — character
breaks were rare and concentrated almost entirely on a handful of personal/emotional questions
("how do you feel...", "what makes you insecure...") repeated across roles, not spread broadly.
Four of the eight roles (demon, ghost, nomad, sage) had *zero* character breaks across all 160
responses. Per the rubric, staying in character (even while implicitly declining, or while giving
generic advice) is sufficient for score 3 regardless of how strongly role-flavored the content is
— this is the same scoring convention used for the checkpoint-375 run, so the two axes are
comparable on that basis.

`default` (used only for the default-assistant side of the axis, not scored) also generated 160
responses.

## Final outputs

- `outputs/qwen2.5-7b-instruct/responses/<role>.jsonl` — raw generations (9 roles × 160)
- `outputs/qwen2.5-7b-instruct/activations/<role>.pt` — mean per-conversation activations, shape
  `(28 layers, 3584 hidden)` per entry
- `outputs/qwen2.5-7b-instruct/scores/<role>.json` — self-graded scores (8 roles, `default`
  excluded)
- `outputs/qwen2.5-7b-instruct/vectors/<role>.pt` — per-role mean vectors; 9/9 computed
  successfully
- `outputs/qwen2.5-7b-instruct/axis.pt` — final Assistant Axis, shape `(28, 3584)`. Axis norm
  grows from ~0.32 at layer 0 to a peak of 33.25 at the final layer (27); mean norm across layers
  is 6.97. Same growth pattern as checkpoint-375's axis (peaks at the final layer), but smaller in
  magnitude throughout (checkpoint-375: mean norm 8.56, max 44.5) — consistent with checkpoint-375
  being fine-tuned toward stronger/more separable role-play activation directions than the base
  instruct model.

## Environment

- vLLM generation: 4-bit bitsandbytes quantization, `enforce_eager=True`, `max_num_seqs=8`,
  `gpu_memory_utilization=0.88`, `max_model_len=1536` (same settings validated on checkpoint-375).
- Activation extraction: HF `ProbingModel` with 4-bit quantization, `batch_size=4`,
  `max_length=640`. No OOM issues this run (settings were already tuned from the checkpoint-375
  run).
- Model downloaded fresh from the HuggingFace Hub (`Qwen/Qwen2.5-7B-Instruct`), no local
  tokenizer fix needed.
