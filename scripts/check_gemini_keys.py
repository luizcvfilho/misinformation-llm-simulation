"""Check each Gemini key independently with one small generation request."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from misinformation_simulation.enums.models import Models  # noqa: E402

KEY_NAMES = ("GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=Models.GEMINI35FlashLite.value)
    args = parser.parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    keys = {name: (os.getenv(name) or "").strip() for name in KEY_NAMES}
    seen = {}
    passed = 0
    print(f"Model: {args.model}. One attempt per distinct key; no automatic failover.", flush=True)
    for name, key in keys.items():
        if not key:
            print(f"{name}: MISSING", flush=True)
            continue
        if key in seen:
            print(f"{name}: DUPLICATE of {seen[key]} (not requested again)", flush=True)
            continue
        seen[key] = name
        print(f"{name}: checking...", flush=True)
        try:
            with genai.Client(
                api_key=key,
                http_options=types.HttpOptions(
                    timeout=30_000,
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            ) as client:
                response = client.models.generate_content(
                    model=args.model,
                    contents="Reply with only the word OK.",
                    config=types.GenerateContentConfig(max_output_tokens=256),
                )
                if not (response.text or "").strip():
                    raise ValueError("The API returned no text; generation was not confirmed.")
        except Exception as error:
            message = str(error)
            for secret in sorted(filter(None, keys.values()), key=len, reverse=True):
                message = message.replace(secret, "[REDACTED]")
            message = " ".join(message.split())
            print(f"{name}: FAIL ({type(error).__name__}) {message}", flush=True)
        else:
            passed += 1
            print(f"{name}: OK (text generation succeeded)", flush=True)
    print(f"Passed: {passed}/{len(KEY_NAMES)} distinct keys.", flush=True)
    return 0 if passed == len(KEY_NAMES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
