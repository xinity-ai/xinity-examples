"""
Streaming benchmark for an OpenAI-compatible chat completions endpoint.

Runs a stepwise concurrency sweep over three input prompt sizes and
reports time to first token (TTFT) and throughput in tokens per
second. Output length is left unbounded by the script and is itself
reported as a measured variable.

Usage:
    uv run python benchmark.py
    uv run python benchmark.py --levels 1,2,4 --repeats 1
    python benchmark.py --scenarios qa_50

Configuration is read from a .env file in the working directory.
See .env.example for the required keys.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI


SCENARIOS: dict[str, tuple[str, str]] = {
    "qa_50": (
        "qa_50.txt",
        "Answer the following question concisely.",
    ),
    "rag_500": (
        "rag_500.txt",
        "Use the provided context documents to answer the question at the end.",
    ),
    "doc_2000": (
        "doc_2000.txt",
        "Read the document below and produce the requested analysis at the end.",
    ),
}

LEVELS_DEFAULT = [1, 2, 4, 8, 16, 32, 64]
REPEATS_DEFAULT = 3
COOLDOWN_DEFAULT = 2.0
TIMEOUT_DEFAULT = 120.0


@dataclass
class RequestResult:
    ok: bool
    ttft: float
    total_time: float
    output_tokens: int
    usage_from_server: bool
    error: str | None


@dataclass
class LevelSummary:
    concurrency: int
    ttft_mean: float
    ttft_p95: float
    out_tokens_mean: float
    total_time_mean: float
    throughput_mean: float
    successes: int
    failures: int
    any_estimated: bool


async def run_one(
    client: AsyncOpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout: float,
) -> RequestResult:
    start = time.perf_counter()
    ttft: float | None = None
    chunk_count = 0
    usage_completion_tokens: int | None = None

    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
            stream_options={"include_usage": True},
            timeout=timeout,
        )
        async for chunk in stream:
            now = time.perf_counter()
            # Some servers send a final chunk that carries usage and no choices.
            if getattr(chunk, "usage", None) is not None:
                usage_completion_tokens = chunk.usage.completion_tokens
            if chunk.choices:
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    # Skip role-only first chunks where content is empty or None.
                    if ttft is None:
                        ttft = now - start
                    chunk_count += 1
    except Exception as exc:
        return RequestResult(
            ok=False,
            ttft=0.0,
            total_time=time.perf_counter() - start,
            output_tokens=0,
            usage_from_server=False,
            error=f"{type(exc).__name__}: {exc}",
        )

    total_time = time.perf_counter() - start
    if ttft is None:
        return RequestResult(
            ok=False,
            ttft=0.0,
            total_time=total_time,
            output_tokens=0,
            usage_from_server=False,
            error="stream produced no content",
        )

    if usage_completion_tokens is not None:
        return RequestResult(
            ok=True,
            ttft=ttft,
            total_time=total_time,
            output_tokens=usage_completion_tokens,
            usage_from_server=True,
            error=None,
        )
    return RequestResult(
        ok=True,
        ttft=ttft,
        total_time=total_time,
        output_tokens=chunk_count,
        usage_from_server=False,
        error=None,
    )


async def run_level(
    client: AsyncOpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    concurrency: int,
    repeats: int,
    timeout: float,
) -> tuple[list[RequestResult], list[float]]:
    all_results: list[RequestResult] = []
    batch_durations: list[float] = []
    for _ in range(repeats):
        batch_start = time.perf_counter()
        results = await asyncio.gather(
            *[
                run_one(client, model, system_prompt, user_prompt, timeout)
                for _ in range(concurrency)
            ]
        )
        batch_durations.append(time.perf_counter() - batch_start)
        all_results.extend(results)
    return all_results, batch_durations


def mean_p95(values: list[float]) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (values[0], values[0])
    mean = statistics.mean(values)
    p95 = statistics.quantiles(values, n=20)[18]
    return (mean, p95)


def summarize(
    concurrency: int,
    results: list[RequestResult],
    batch_durations: list[float],
) -> LevelSummary:
    ok = [r for r in results if r.ok]
    failures = len(results) - len(ok)
    if not ok:
        return LevelSummary(
            concurrency=concurrency,
            ttft_mean=0.0,
            ttft_p95=0.0,
            out_tokens_mean=0.0,
            total_time_mean=0.0,
            throughput_mean=0.0,
            successes=0,
            failures=failures,
            any_estimated=False,
        )

    ttft_mean, ttft_p95 = mean_p95([r.ttft for r in ok])
    out_tokens_mean = statistics.mean([r.output_tokens for r in ok])
    total_time_mean = statistics.mean([r.total_time for r in ok])

    # Aggregate throughput per batch: sum of output tokens for the batch
    # divided by the batch wall clock. Then averaged across batches.
    per_batch_throughput: list[float] = []
    idx = 0
    for batch_seconds in batch_durations:
        batch_results = results[idx : idx + concurrency]
        idx += concurrency
        batch_tokens = sum(r.output_tokens for r in batch_results if r.ok)
        if batch_seconds > 0 and batch_tokens > 0:
            per_batch_throughput.append(batch_tokens / batch_seconds)
    throughput_mean = (
        statistics.mean(per_batch_throughput) if per_batch_throughput else 0.0
    )

    return LevelSummary(
        concurrency=concurrency,
        ttft_mean=ttft_mean,
        ttft_p95=ttft_p95,
        out_tokens_mean=out_tokens_mean,
        total_time_mean=total_time_mean,
        throughput_mean=throughput_mean,
        successes=len(ok),
        failures=failures,
        any_estimated=any(not r.usage_from_server for r in ok),
    )


def print_table(
    scenario: str,
    prompt_token_estimate: int,
    rows: list[LevelSummary],
) -> None:
    header_cols = [
        ("concurrency", 11),
        ("TTFT mean (s)", 14),
        ("TTFT p95 (s)", 13),
        ("out tokens avg", 15),
        ("total time avg (s)", 19),
        ("throughput (tok/s)", 19),
    ]

    print()
    print(
        f"Scenario: {scenario}  (input ~{prompt_token_estimate} tokens, output unbounded)"
    )
    header = "  ".join(name.rjust(width) for name, width in header_cols)
    print(header)
    print("-" * len(header))

    any_estimated = False
    for r in rows:
        if r.successes == 0:
            line = (
                f"{r.concurrency:>11}  "
                f"{'n/a':>14}  {'n/a':>13}  {'n/a':>15}  {'n/a':>19}  {'n/a':>19}"
            )
            print(line + f"   [{r.failures} failures]")
            continue
        marker = "*" if r.any_estimated else " "
        if r.any_estimated:
            any_estimated = True
        print(
            f"{r.concurrency:>11}  "
            f"{r.ttft_mean:>14.3f}  "
            f"{r.ttft_p95:>13.3f}  "
            f"{r.out_tokens_mean:>14.1f}{marker}  "
            f"{r.total_time_mean:>19.3f}  "
            f"{r.throughput_mean:>19.2f}"
        )
        if r.failures:
            print(f"{'':>11}  [{r.failures} failures at this level]")

    if any_estimated:
        print(
            "* output tokens estimated from streamed chunks "
            "(server did not return a usage block)"
        )


def parse_levels(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_scenarios(value: str) -> list[str]:
    items = [x.strip() for x in value.split(",") if x.strip()]
    unknown = [x for x in items if x not in SCENARIOS]
    if unknown:
        raise SystemExit(
            f"Unknown scenario(s): {unknown}. Available: {list(SCENARIOS)}"
        )
    return items


def load_prompt(prompts_dir: Path, filename: str) -> str:
    path = prompts_dir / filename
    if not path.exists():
        raise SystemExit(f"Prompt fixture not found: {path}")
    return path.read_text(encoding="utf-8")


async def main_async(args: argparse.Namespace) -> int:
    load_dotenv()
    api_key = os.environ.get("API_KEY")
    base_url = os.environ.get("BASE_URL")
    model = os.environ.get("MODEL")
    missing = [n for n, v in [("API_KEY", api_key), ("BASE_URL", base_url), ("MODEL", model)] if not v]
    if missing:
        print(
            f"Missing required environment variables: {missing}. "
            f"Copy .env.example to .env and fill in the values.",
            file=sys.stderr,
        )
        return 1

    prompts_dir = Path(__file__).resolve().parent / "prompts"

    scenarios = parse_scenarios(args.scenarios) if args.scenarios else list(SCENARIOS)
    levels = parse_levels(args.levels) if args.levels else LEVELS_DEFAULT

    print(f"Endpoint: {base_url}")
    print(f"Model: {model}")
    print(f"Scenarios: {', '.join(scenarios)}")
    print(f"Concurrency levels: {', '.join(str(x) for x in levels)}")
    print(f"Repeats per level: {args.repeats}")

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    try:
        for scenario in scenarios:
            filename, system_prompt = SCENARIOS[scenario]
            user_prompt = load_prompt(prompts_dir, filename)
            prompt_token_estimate = int(scenario.rsplit("_", 1)[-1])

            print()
            print(f"Warming up scenario {scenario} ...")
            warm = await run_one(
                client, model, system_prompt, user_prompt, args.timeout
            )
            if not warm.ok:
                print(f"Warm up failed: {warm.error}", file=sys.stderr)
                # Continue anyway; failures will surface in the level runs too.

            rows: list[LevelSummary] = []
            for i, concurrency in enumerate(levels):
                print(
                    f"Running concurrency={concurrency}, "
                    f"repeats={args.repeats} ..."
                )
                results, batch_durations = await run_level(
                    client,
                    model,
                    system_prompt,
                    user_prompt,
                    concurrency,
                    args.repeats,
                    args.timeout,
                )
                rows.append(summarize(concurrency, results, batch_durations))
                if i < len(levels) - 1 and args.cooldown > 0:
                    await asyncio.sleep(args.cooldown)

            print_table(scenario, prompt_token_estimate, rows)
    finally:
        await client.close()

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Streaming benchmark for an OpenAI-compatible endpoint."
    )
    parser.add_argument(
        "--scenarios",
        default="",
        help="Comma separated scenario names. Default: all three.",
    )
    parser.add_argument(
        "--levels",
        default="",
        help="Comma separated concurrency levels. Default: 1,2,4,8,16,32,64.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=REPEATS_DEFAULT,
        help=f"Repeats per concurrency level. Default: {REPEATS_DEFAULT}.",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=COOLDOWN_DEFAULT,
        help=(
            f"Seconds to sleep between concurrency levels. "
            f"Default: {COOLDOWN_DEFAULT}."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=TIMEOUT_DEFAULT,
        help=f"Per request timeout in seconds. Default: {TIMEOUT_DEFAULT}.",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
