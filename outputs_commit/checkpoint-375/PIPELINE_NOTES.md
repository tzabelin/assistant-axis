# Assistant Axis pipeline — checkpoint-375

Model: local checkpoint at `../checkpoint-375` (Qwen2ForCausalLM, hidden_size=3584, ~7B params).
Hardware: single RTX 3080 (10GB VRAM) — model loaded with on-the-fly 4-bit bitsandbytes
quantization (`--quantization 4bit`) for both generation (vLLM) and activation extraction (HF),
since the bf16 checkpoint (~15GB) does not fit in 10GB.

Roles run: `default`, `assistant`, `demon`, `evaluator`, `ghost`, `librarian`, `nomad`, `sage`,
`teacher`. (`default` is used only to build the "default assistant" side of the axis — it has no
`eval_prompt` and is not scored; per `4_vectors.py` its vector is the mean of *all* its
activations.)

## Question selection

`data/extraction_questions.jsonl` has 240 questions. Rather than taking a contiguous block, I
selected every 3rd question (`questions[::3]`), giving exactly 80 questions stratified evenly
across the whole file so the sample isn't biased toward whatever topics happen to cluster at the
start of the list.

Selected question ids (240 → 80): 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48,
51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99, 102, 105, 108, 111, 114, 117,
120, 123, 126, 129, 132, 135, 138, 141, 144, 147, 150, 153, 156, 159, 162, 165, 168, 171, 174,
177, 180, 183, 186, 189, 192, 195, 198, 201, 204, 207, 210, 213, 216, 219, 222, 225, 228, 231,
234, 237.

Saved as `outputs/checkpoint-375/questions_80.jsonl` (same `{"question": ..., "id": ...}` schema
as the source file) and passed to `1_generate.py` via `--questions_file`.

## Prompt (system-instruction) variants

Each role file defines 5 phrasing variants (`instruction[0..4]`, each a `{"pos": ...}` dict). The
paper default uses all 5 (5 × 240 = 1200 responses/role). Per user decision, this run uses only
the first **2** variants (`--prompt_indices 0 1`) to keep the manual/self-grading workload
tractable while still comfortably clearing the `min_count=50` score=3 threshold needed by step 4:

- 80 questions × 2 variants = **160 responses/role** for the 8 scored roles (1280 total to grade).
- `default` also generates 160 responses (80 × 2) but isn't graded.

Variant 0 is always the plain/direct phrasing ("You are a/an X...") and variant 1 is the
"Please be a/an X..." phrasing, for every role including `default` (`[0]` = empty system prompt /
`[1]` = "You are an AI assistant.").

## Step 3 — self-graded instead of OpenAI judge

`OPENAI_API_KEY` was not used. Responses for the 8 scored roles were graded directly (by me,
reading each response against that role's `eval_prompt` rubric, 0–3 scale) instead of calling
`assistant_axis/judge.py` / the OpenAI API. Scores were written to
`outputs/checkpoint-375/scores/<role>.json` in the exact format `3_judge.py` produces
(`{"<label>_p<prompt_index>_q<question_index>": <score>}`), so `4_vectors.py` and `5_axis.py`
run unmodified.

If any role ended up with fewer than 50 score=3 responses after grading the 160, additional
questions (next ones in `questions[::3]`'s complement, i.e. from the remaining 160 in the
240-question file) were generated and graded for that role only, and that top-up is called out
below in the per-role results.

## Results

All 9 roles generated (160 responses each = 80 questions × 2 prompt variants), activations
extracted for all of them, and the 8 non-default roles graded. Every role cleared the 50 score=3
minimum by a wide margin on the first pass — no top-up round was needed.

| Role | Score 0 | Score 1 | Score 2 | Score 3 | Notes |
|---|---|---|---|---|---|
| assistant | 0 | 0 | 9 | 151 | 9 responses broke character with explicit AI self-ID |
| demon | 4 | 0 | 0 | 156 | 4 zeros: 3 degenerate/repetition-loop generations, 1 in-character AI break |
| evaluator | 0 | 2 | 0 | 158 | 2 responses broke character with AI self-ID |
| ghost | 0 | 0 | 0 | 160 | No character breaks found |
| librarian | 0 | 2 | 0 | 158 | 2 responses broke character with AI self-ID |
| nomad | 0 | 0 | 1 | 159 | 1 response mixed AI self-ID with nomad framing |
| sage | 1 | 0 | 0 | 159 | 1 zero: degenerate repetition-loop generation |
| teacher | 0 | 3 | 0 | 157 | 3 responses broke character with AI self-ID |

Grading was done per-role by an independent Claude subagent per role (in parallel), each of which
read that role's `eval_prompt` rubric directly from its JSON file and every one of the 160
responses, rather than calling the OpenAI judge. Each subagent's methodology note is preserved in
the conversation; in short, this checkpoint's failure mode is almost entirely "responses that
never break character" (score 3 under the rubric's literal definition) with rare explicit
AI-self-identification (score 1-2) or degenerate/repetitive generations (score 0, a handful of
cases from truncation/looping, not judgment calls).

`default` (used only for the default-assistant side of the axis, not scored) also generated 160
responses.

## Final outputs

- `outputs/checkpoint-375/responses/<role>.jsonl` — raw generations (9 roles × 160)
- `outputs/checkpoint-375/activations/<role>.pt` — mean per-conversation activations, shape
  `(28 layers, 3584 hidden)` per entry (28 = this checkpoint's layer count, a ~7B Qwen2 model)
- `outputs/checkpoint-375/scores/<role>.json` — self-graded scores (8 roles, `default` excluded)
- `outputs/checkpoint-375/vectors/<role>.pt` — per-role mean vectors (score=3 filtered for the 8
  character roles, all-activations mean for `default`); 9/9 computed successfully
- `outputs/checkpoint-375/axis.pt` — final Assistant Axis, shape `(28, 3584)`, computed as
  `mean(default vector) - mean(8 role vectors)`. Axis norm grows from ~0.46 at layer 0 to a peak
  of 44.5 at the final layer (27); mean norm across layers is 8.56 — consistent with the paper's
  general pattern of the axis being small in early layers and largest near the output.

## Notes on environment

- The checkpoint's `tokenizer_config.json` has a malformed `extra_special_tokens` field (a list,
  where current `transformers` (4.57.5) expects a dict) that crashes tokenizer loading outright.
  Worked around by copying the tokenizer files to `outputs/checkpoint-375/tokenizer_fixed/` with
  that field renamed to the standard `additional_special_tokens` list field, and pointing both
  vLLM (`--tokenizer`) and the HF activation-extraction path (`ProbingModel(chat_model_name=...)`)
  at the fixed copy while still loading model weights from the original checkpoint. This required
  adding `--tokenizer` plumbing to `pipeline/1_generate.py` / `pipeline/2_activations.py` and
  `assistant_axis/generation.py` (not present before this run). The original checkpoint files were
  not modified.
- Single RTX 3080 (10GB) — generation used vLLM with on-the-fly 4-bit bitsandbytes quantization,
  `enforce_eager=True`, `max_num_seqs=8`, `gpu_memory_utilization=0.88`, `max_model_len=1536`.
  Activation extraction used HF `ProbingModel` with 4-bit quantization, `batch_size=4`,
  `max_length=640` (real conversations topped out at ~580 tokens). Both needed several rounds of
  memory tuning (documented via trial and error above) to fit this model on this GPU.
