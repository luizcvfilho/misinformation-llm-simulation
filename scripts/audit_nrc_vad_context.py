from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from misinformation_simulation.audits.nrc_vad_context import run_nrc_context_audit  # noqa: E402
from misinformation_simulation.text_metrics.nrc_vad import NRC_VAD_DOWNLOAD_URL  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate NRC VAD v2.1 on paired English contexts."
    )
    parser.add_argument(
        "--input", type=Path, default=PROJECT_ROOT / "data/synthetic/vad_context_contrast_news.csv"
    )
    parser.add_argument(
        "--lexicon", type=Path, default=PROJECT_ROOT / ".cache/nrc_vad/NRC-VAD-Lexicon-v2.1.zip"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the official archive if absent (noncommercial research/education).",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=PROJECT_ROOT
        / "output/audit/VADContextContrastAudit/vad_context_contrast_scored.csv",
    )
    parser.add_argument("--without-baseline", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "output/audit/NRCVADContextContrastAudit"
    )
    args = parser.parse_args()
    if args.download and not args.lexicon.exists():
        with urlopen(NRC_VAD_DOWNLOAD_URL, timeout=60) as response:
            payload = response.read()
        args.lexicon.parent.mkdir(parents=True, exist_ok=True)
        args.lexicon.write_bytes(payload)
    if not args.lexicon.exists():
        parser.error("Lexicon missing. Use --download or --lexicon with the official v2.1 ZIP/TSV.")
    manifest = run_nrc_context_audit(
        args.input,
        args.lexicon,
        args.output_dir,
        baseline_path=None if args.without_baseline else args.baseline,
    )
    print(f"Scored {manifest['documents']} documents ({manifest['pairs']} pairs).")
    print(f"Token coverage: {manifest['weighted_token_coverage']:.1%}")
    print(f"Findings: {(args.output_dir / 'vad_context_contrast_findings.md').resolve()}")


if __name__ == "__main__":
    main()
