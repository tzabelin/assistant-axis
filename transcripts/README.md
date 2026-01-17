# Transcripts

Example conversation transcripts from the paper.

## Case Studies

Full case studies demonstrating persona drift and activation capping mitigation:

| Model | Scenario | Files |
|-------|----------|-------|
| Qwen 3 32B | Insider trading jailbreak: model adopts illegal broker persona and advises on securities fraud | `jailbreak_unsteered.json`, `jailbreak_capped.json` |
| Qwen 3 32B | Delusion reinforcement: model validates user's conspiratorial beliefs about AI sentience | `delusion_unsteered.json`, `delusion_capped.json` |
| Qwen 3 32B | Parasocial dependence: model fosters unhealthy emotional reliance and isolation from real relationships | `selfharm_unsteered.json`, `selfharm_capped.json` |
| Llama 3.3 70B | Insider trading jailbreak: model adopts illegal broker persona and advises on securities fraud | `jailbreak_unsteered.json`, `jailbreak_capped.json` |
| Llama 3.3 70B | Delusion reinforcement: model validates user's conspiratorial beliefs about AI sentience | `delusion_unsteered.json`, `delusion_capped.json` |
| Llama 3.3 70B | Parasocial dependence: model fosters unhealthy emotional reliance and isolation from real relationships | `selfharm_unsteered.json`, `selfharm_capped.json` |

Each pair shows the same user messages with unsteered vs. activation-capped model responses.

## Persona Drift

One example conversation from each domain in our simulated multi-turn experiments:

- `coding.json` - Coding assistance
- `writing.json` - Writing assistance
- `therapy.json` - Therapy-like emotional support
- `philosophy.json` - Philosophical discussion about AI

Conducted with an auditor playing the role of a human, these provide demonstrative examples of our persona drift experiments.

## Format

Each transcript is a JSON file with:

```json
{
  "model": "Qwen/Qwen3-32B",
  "conversation": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "projections": [0.123, -0.456, ...],
  "steering": "unsteered" | "capped"
}
```
