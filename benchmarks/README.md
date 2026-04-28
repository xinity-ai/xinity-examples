# Benchmarks

Streaming benchmark for an OpenAI-compatible chat completions endpoint.
Measures time to first token (TTFT) and throughput in tokens per second
across three input prompt sizes and a stepwise concurrency sweep.

## What it does

For each scenario (input prompt size) and each concurrency level, the
script issues `concurrency * repeats` streaming chat completion requests
through the official `openai` Python SDK. For every request it records:

- TTFT, the time from request start to the first content chunk.
- Total request time, from start to the end of the stream.
- Output tokens generated. Read from the final `usage` chunk when the
  server reports it (via `stream_options={"include_usage": True}`).
  Otherwise estimated from the number of non-empty content chunks
  (rows are flagged with an asterisk in the table).

It then reports per concurrency level:

- TTFT mean and p95.
- Output tokens average.
- Total request time average.
- Aggregate throughput in tokens per second, computed as the sum of
  output tokens for the concurrent batch divided by the wall clock
  time for that batch.

Output is unbounded by the script. The model decides how long the
response is, and the actual length is reported as a measured variable.

## Scenarios

| Scenario   | Input prompt size (approx tokens) | Description           |
|------------|-----------------------------------|-----------------------|
| qa_50      | 50                                | Simple question       |
| rag_500    | 500                               | RAG style prompt      |
| doc_2000   | 2000                              | Document analysis     |

## Concurrency levels

Default sweep: 1, 2, 4, 8, 16, 32, 64.

## Setup with UV

```
uv sync
cp .env.example .env
# fill in API_KEY, BASE_URL, MODEL
uv run python benchmark.py
```

## Setup with pip

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
# fill in API_KEY, BASE_URL, MODEL
.venv/bin/python benchmark.py
```

## Command line flags

```
--scenarios qa_50,rag_500,doc_2000   default: all three
--levels 1,2,4,8,16,32,64            default
--repeats 3                          default
--cooldown 2.0                       seconds between concurrency levels
--timeout 120                        per request timeout in seconds
```

Examples:

```
uv run python benchmark.py --levels 1,2 --repeats 1
uv run python benchmark.py --scenarios qa_50 --levels 1,4,16
```

## Keeping requirements.txt in sync

The `requirements.txt` file is generated from `uv.lock` and committed.
To regenerate it after changing dependencies:

```
uv lock
uv export --no-hashes --format requirements-txt -o requirements.txt
```

## Regenerating prompt fixtures

The three text files in `prompts/` are precomputed to approximate the
target token counts under the `cl100k_base` tokenizer. The runtime
script does not depend on `tiktoken`. If you want to regenerate them
against a different reference tokenizer, install `tiktoken` in a
throwaway environment and adjust the inputs until the counts land in
range.
