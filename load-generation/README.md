# Load Generator

Highly parallel load generator for LLM serving workloads. It samples
conversations from prompt templates (single-round and multi-round), dispatches
them at a controlled rate against an OpenAI-compatible endpoint (e.g. vLLM),
and records per-request and aggregate statistics.

## How it works

- **Rate slicing**: the requested `samples_per_second` rate is sliced into
  10 ms ticks. Each tick releases a batch of new conversations, with the
  fractional part of the rate spread evenly so the long-term average matches
  the requested rate exactly.
- **Sampling**: prompts come from `PromptSampler`, driven by a YAML config
  that mixes template sources by ratio. Multi-round templates yield one user
  message per round.
- **Multi-round replay**: each round is sent one message at a time; the
  assistant reply is awaited and appended to the history before the next
  message is sent.
- **Retries**: a failed message exchange is retried with exponential backoff
  (`retry_backoff_base_seconds * 2^(attempt-1)`), up to `max_retries`
  attempts; after that the request is marked failed.
- **Single active run**: only one run may be active at a time; concurrent
  requests are rejected with HTTP 409.

## Setup

```bash
cd load-generation
uv sync          # or: pip install fastapi 'uvicorn[standard]' httpx pydantic pyyaml
```

## Configuration

Copy `config.example.yaml` and adjust it. The config only controls **how the
generator behaves** — prompt sampling, retries, timeouts, dispatch
granularity. All **endpoint information** (IP address, port, model) is
supplied per run in the `POST /load` request body.

```yaml
seed: 42
buzzwords_file: templates/buzzwords.txt

sources:
  - name: interactive_chat
    type: single_round                # one user message per request
    templates_dir: templates/question_prompts
    ratio: 0.8                        # 80% of requests
  - name: multi_turn_qa
    type: multi_round                 # one user message per round
    templates_dir: templates/question_prompts_multi
    ratio: 0.2                        # 20% of requests

behavior:
  max_retries: 3
  retry_backoff_base_seconds: 0.5
  request_timeout_seconds: 60

dispatch:
  tick_interval_seconds: 0.01         # batch release period
```

The server reads `LOAD_GENERATOR_CONFIG` if set, otherwise
`config.example.yaml` next to `server.py`.

## Running the server

```bash
uvicorn server:app --host 0.0.0.0 --port 9000
# or with a custom config:
LOAD_GENERATOR_CONFIG=/path/to/config.yaml uvicorn server:app --port 9000
```

## API

### `POST /load` — start a run

```bash
curl -X POST http://localhost:9000/load \
  -H 'Content-Type: application/json' \
  -d '{"ip_address": "127.0.0.1", "port": 8000, "model": "Qwen/Qwen2.5-0.5B-Instruct", "samples_per_second": 3, "duration_seconds": 30}'
```

`port` (default 8000) and `model` (default `"default"`) are optional.
Returns `202 Accepted` with the echoed config, or `409 Conflict` if a run is
already in progress. `422` means invalid input (bad IP, non-positive
rate/duration).

### `GET /runs` — inspect runs and statistics

```bash
curl http://localhost:9000/runs
```

Each entry contains the run config, a `running` flag, and stats (computed
live while the run is active, final once finished):

| field                   | meaning                                             |
|-------------------------|-----------------------------------------------------|
| `duration_seconds`      | wall-clock from run start to finish (or now)        |
| `requests_total`        | finished requests so far                            |
| `requests_succeeded` / `requests_failed` | outcome counts               |
| `success_rate` / `failure_rate`          | 0–1                          |
| `prompt_tokens` / `completion_tokens`    | totals reported by the endpoint |
| `avg_tokens_per_second` | mean per-request completion throughput               |
| `tokens_per_second`     | completion tokens ÷ run duration                     |

### `GET /health`

```bash
curl http://localhost:9000/health   # {"status": "ok"}
```

## Library usage

```python
import asyncio
from core import GeneratorSettings, LoadConfig, LoadGenerator, PromptSampler

async def main():
    gen = LoadGenerator(
        sampler=PromptSampler.from_recipe("config.example.yaml"),
        settings=GeneratorSettings.from_file("config.example.yaml"),
    )
    run = await gen.start_load(
        LoadConfig(
            ip_address="127.0.0.1",
            samples_per_second=3,
            duration_seconds=30,
            port=8000,                            # optional, default 8000
            model="Qwen/Qwen2.5-0.5B-Instruct",   # optional
        )
    )
    await run.task            # wait for completion
    print(run.stats)          # aggregate statistics
    print(run.results)        # per-request RequestResult records

asyncio.run(main())
```

## Testing

```bash
uv run pytest          # API tests (tests/test_server.py)
```
