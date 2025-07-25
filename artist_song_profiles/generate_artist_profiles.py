#!/usr/bin/env python3
"""
Asynchronously generate audio descriptions for songs using the
OpenAI Responses API + web_search_preview, with concurrency derived
from a single RATE_LIMIT_RPM constant.

Usage:
  python generate_artist_profiles.py -i selected_artists_v0.json \
                                -o artist_profiles_v0.jsonl \
                                -n 100          # optional slice for testing
                                -p prompts/artist_profile_v0.txt
"""
import argparse, asyncio, json, math, time
from pathlib import Path
from typing import List
from pydantic import BaseModel

from openai import AsyncOpenAI, RateLimitError, APIError   # pip install --upgrade openai
from tqdm import tqdm                                      # progress visualization

# ──────────────────────────────── RATE-LIMIT SETTINGS ─────────────────────────
RATE_LIMIT_RPM   = 500                  # <-- edit this if your quota changes
SAFETY_FACTOR    = 0.80                 # 20 % head-room
RPS              = RATE_LIMIT_RPM / 60  # requests per second
MAX_CONCURRENCY  = max(1, int(RPS * SAFETY_FACTOR))
MAX_RETRIES      = 3
RETRY_BACKOFF_SEC = max(1, int(60 / RATE_LIMIT_RPM * 10))  # scales w/ quota

# ──────────────────────────────── ARGPARSE ────────────────────────────────
parser = argparse.ArgumentParser(
    description="Batch-generate audio descriptions for a JSON collection."
)
parser.add_argument("-i", "--input",  required=True,
                    help="Path to input artists JSON file (should be a list of artist names)")
parser.add_argument("-o", "--output", required=True,
                    help="Path to output JSONL file")
parser.add_argument("-n", "--num_entries", type=int, default=None,
                    help="Process only the first N entries (testing)")
parser.add_argument("-l", "--log", default=None,
                    help="Path to raw API response log file (defaults to <output>.raw.jsonl)")
parser.add_argument("-p", "--prompt", required=True,
                        help="Path to prompt template txt file (variable {{artist}} will be replaced with the artist name)")

cli_args = parser.parse_args()

# ─────────────────────────────── PROMPT TEMPLATE ──────────────────────────────
PROMPT_TEMPLATE = Path(cli_args.prompt).read_text(encoding="utf-8").strip()

# ─────────────────────────────── OUTPUT SCHEMA ──────────────────────────────
class ArtistProfile(BaseModel):
    artist: str
    familiar: bool
    musical_style: str | None
    lyrical_themes: str | None
    mood: str | None

# ────────────────────────────── HELPER FUNCTIONS ──────────────────────────────
def build_prompt(artist: str) -> str:
    return (PROMPT_TEMPLATE
            .replace("{{artist}}", artist))

# ──────────────────────────────── ASYNC WORKER ────────────────────────────────
client = AsyncOpenAI()                      # requires OPENAI_API_KEY env-var
sem    = asyncio.Semaphore(MAX_CONCURRENCY)

async def generate_profile(artist: str) -> dict:
    prompt = build_prompt(artist)
    prompt = prompt               # save even on error

    for attempt in range(1, MAX_RETRIES + 1):
        async with sem:                    # limits concurrent requests
            try:
                resp = await client.responses.parse(
                    model  = "gpt-4.1",           # or any model that supports tools
                    input  = prompt,
                    text_format=ArtistProfile,
                )
                artist_profile = resp.output_parsed
                # Return both the parsed profile and the raw response for logging
                return artist_profile.model_dump(exclude_none=True), resp.model_dump(exclude_none=True)  # success → done

            except (RateLimitError, APIError) as err:
                if attempt == MAX_RETRIES:
                    raise
                wait = RETRY_BACKOFF_SEC * attempt
                print(f"[retry {attempt}/{MAX_RETRIES}] {err} → sleeping {wait}s")
                await asyncio.sleep(wait)

# ──────────────────────────────── MAIN DRIVER ────────────────────────────────
async def main(args) -> None:
    with open(args.input, "r") as f:
        artists = json.load(f)

    if args.num_entries:                   # optional slice for quick tests
        artists = artists[: args.num_entries]

    # Determine log file path (defaults to <output>.raw.jsonl if not provided)
    log_path = Path(args.log) if args.log else Path(f"{args.output}.raw.jsonl")

    # Regular file handles are synchronous; use standard 'with' instead of 'async with'.
    with Path(args.output).open("w", encoding="utf-8") as sink, \
         log_path.open("w", encoding="utf-8") as raw_sink:
        tasks = [asyncio.create_task(generate_profile(artist)) for artist in artists]

        # Real-time progress bar
        with tqdm(total=len(tasks), unit="artist") as pbar:
            for coro in asyncio.as_completed(tasks):
                artist_profile, raw_resp = await coro
                sink.write(json.dumps(artist_profile, ensure_ascii=False) + "\n")
                raw_sink.write(json.dumps(raw_resp, ensure_ascii=False) + "\n")
                pbar.update(1)

if __name__ == "__main__":
    t0 = time.perf_counter()
    asyncio.run(main(cli_args))
    elapsed = time.perf_counter() - t0
    print(f"\nAll done in {elapsed:.1f}s with "
          f"{MAX_CONCURRENCY}-way concurrency (rate ≤ {RATE_LIMIT_RPM} RPM)")
