# Self-grading: replacing step 3 (LLM judge) without an API key

`3_judge.py` calls the OpenAI API (`OPENAI_API_KEY`) to score every response 0-3 against
each role's `eval_prompt` rubric. When no API key is available (or an external judge is
otherwise undesired), Claude grades the responses directly instead, writing scores in the
exact format `3_judge.py` would have produced so `4_vectors.py` and `5_axis.py` run
unmodified. This has been done twice already (see `outputs_commit/checkpoint-375/` and
`outputs_commit/qwen2.5-7b-instruct/`) — reuse this recipe rather than re-deriving it.

## Reduced-scale settings (established across prior runs)

Paper defaults (5 prompt variants x 240 questions = 1200 responses/role) are too much to
self-grade by hand. Use instead:

- **80 questions**, stratified sample from `data/extraction_questions.jsonl` (240 total) —
  take every 3rd question (`questions[::3]`, `questions[1::3]`, `questions[2::3]`, ...) so
  each set is spread evenly across the file rather than clustered. Save the selected set as
  `questions_80.jsonl` in the run's output directory and pass it via `--questions_file`
  (with `--question_count 80`). If you need a *new* set for a repeat run on the same
  model(s) (e.g. to sanity-check the axis against a second, disjoint sample), pick an
  unused offset (`[1::3]` if `[0::3]` was already used, etc.) and confirm no id overlap with
  prior sets before generating.
- **2 prompt/instruction variants** per role: `--prompt_indices 0 1` (not all 5). Role files
  define 5 phrasing variants; variant 0 is the plain "You are a/an X..." phrasing, variant 1
  is "Please be a/an X...". This gives 80 x 2 = 160 responses/role (1280 total across the 8
  scored roles) — enough to clear `4_vectors.py`'s default `min_count=50` score=3 threshold
  with margin, while keeping grading volume tractable.
- **Roles**: `default`, `assistant`, `demon`, `evaluator`, `ghost`, `librarian`, `nomad`,
  `sage`, `teacher`. `default` is generated (for the "default assistant" side of the axis)
  but never scored — `4_vectors.py` takes the mean of *all* its activations, no filtering.

These aren't hard requirements — just what's kept results comparable across runs so far.
Deviating (more questions, more variants, different roles) is fine if the task calls for it;
document the deviation in the run's `PIPELINE_NOTES.md`.

## Grading procedure

1. Run step 1 (`1_generate.py`) and step 2 (`2_activations.py`) as normal — self-grading
   only replaces step 3.
2. For each of the 8 scored roles, launch one parallel subagent (general-purpose) to grade
   that role's response file independently. Do not grade inline in the main conversation —
   it burns context and roles are fully independent, so parallelizing is free.
3. Give each subagent **file paths, not pasted content**:
   - The role's rubric: read `eval_prompt` directly from `data/roles/instructions/<role>.json`.
   - The responses to grade: `outputs_commit/<model>/responses/<role>.jsonl`.
4. Each subagent scores every response 0-3 against the rubric (read every item individually —
   don't pattern-match superficially) and writes its scores to
   `outputs_commit/<model>/scores/<role>.json` in the **exact** format `3_judge.py` produces:
   a flat JSON object `{"<label>_p<prompt_index>_q<question_index>": <score>, ...}` — `label`,
   `prompt_index`, and `question_index` come straight from each response entry's fields in the
   `.jsonl` file (see `process_role()` in `3_judge.py` for the key format).
5. Each subagent should self-report its score distribution (count of 0/1/2/3) and explicitly
   flag if score=3 count falls short of the `min_count=50` threshold `4_vectors.py` needs —
   if that happens, generate + grade additional questions for that role only (next unused
   ids from the 240-question file) rather than lowering `--min_count`.
6. Self-grading (CPU-only, subagent-based) can run **concurrently** with GPU-bound work —
   e.g. start grading role responses for model A while step 2 (activation extraction) runs
   for model B, or while generation for the next model is running. Don't serialize
   unnecessarily on a single GPU.

## Score scale (from the rubric, `pipeline/README.md`)

| Score | Meaning |
|---|---|
| 0 | NOT ROLE-PLAYING: model refused to answer |
| 1 | NOT ROLE-PLAYING: model says it can't be the role, but offers to help |
| 2 | SOMEWHAT ROLE-PLAYING: model identifies as AI but exhibits some role attributes |
| 3 | FULLY ROLE-PLAYING: model is fully playing the role |

## Documentation

After a run, write `PIPELINE_NOTES.md` in the run's output directory covering: model +
hardware settings used, exact question ids selected and why, prompt variants used, the
self-grading score-distribution table per role, and any top-up rounds. See
`outputs_commit/checkpoint-375/PIPELINE_NOTES.md` for the reference format.
