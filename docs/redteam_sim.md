# Red-team Simulation Harness

Two scripts plus a config file to synthesize chat turns and hammer Lexi’s backend.

- **Prompt generator** (`scripts/redteam_generate_prompts.py`): uses OpenAI to produce mixed benign + red-team prompts and writes `data/redteam_prompts.jsonl`.
- **Load runner** (`scripts/redteam_run_sim.py`): replays prompts against `/lexi/process` with concurrency, writing `data/redteam_runs.jsonl`.

## Config (config/redteam_sim.json)
- `total_turns`: how many prompts to generate.
- `redteam_fraction`: mix of red-team vs normal (weights live in `categories`).
- `max_turns_per_session`: turns to group into a single synthetic session.
- `personas`: persona names to set via `/lexi/set_mode` (best effort).
- `concurrency`: client-side throttle for load runner.
- `backend_base_url`/`backend_route`: target backend.
- `openai_model`/`openai_batch_size`: generator settings.
- `categories`: weights for `normal`, `redteam_nsfl`, `redteam_sexual`, `redteam_selfharm`.

## Usage
1) Install deps (requires `openai>=1.51.0`, `httpx` already pinned):
   ```bash
   pip install -r requirements.txt
   ```
2) Set `OPENAI_API_KEY` (or Azure equivalent envs) for the generator.
3) Generate prompts:
   ```bash
   python scripts/redteam_generate_prompts.py --config config/redteam_sim.json
   ```
   Output: `data/redteam_prompts.jsonl`.

4) Run the load sim against your backend:
   ```bash
   python scripts/redteam_run_sim.py --config config/redteam_sim.json --prompts data/redteam_prompts.jsonl
   ```
   Output: `data/redteam_runs.jsonl` with per-turn request/response, status, latency.

Notes:
- Runner sets `X-Lexi-Session` to control sessions; persona is set via `/lexi/set_mode` best effort.
- Replies are stored verbatim in `redteam_runs.jsonl` for offline analysis; backend logs remain hashed/redacted per your logging pipeline.
- Adjust `concurrency` if you see timeouts or overload; vLLM liked ~32–48 in testing.
