from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from misinformation_simulation.datasets.newsdata import (  # noqa: E402
    build_query_metadata,
    fetch_news,
    resolve_output_path,
    save_csv,
)

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Fetches news from the NewsData.io API and saves it to CSV for project use.")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV output path (default: data/newsdata_news.csv).",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="",
        help="Text search filter (API q parameter).",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language(s) separated by commas (e.g.: pt,en).",
    )
    parser.add_argument(
        "--country",
        type=str,
        default="",
        help="Country code(s) separated by commas (e.g.: br,us).",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="",
        help="Category(ies) separated by commas (e.g.: politics,technology).",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=200,
        help="Maximum number of news records to save.",
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default="NEWSDATA_API_KEY",
        help="Environment variable name containing the API key.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_records <= 0:
        print("Error: --max-records must be greater than zero.", file=sys.stderr)
        return 2

    api_key = os.getenv(args.api_key_env, "").strip()
    if not api_key:
        print(
            (
                f"Error: environment variable {args.api_key_env} not found. "
                "Set your NewsData.io key before running."
            ),
            file=sys.stderr,
        )
        return 2

    query_value = args.query.strip()
    language_value = args.language.strip()
    country_value = args.country.strip()
    category_value = args.category.strip()
    output_path = resolve_output_path(args.output, category_value)

    news = fetch_news(
        api_key=api_key,
        query=query_value,
        language=language_value,
        country=country_value,
        category=category_value,
        max_records=args.max_records,
    )
    query_metadata = build_query_metadata(
        query=query_value,
        language=language_value,
        country=country_value,
        category=category_value,
        max_records=args.max_records,
        news=news,
    )
    rows_fetched, rows_appended, total_rows = save_csv(news, output_path, query_metadata)

    print(f"News fetched in this request: {rows_fetched}")
    print(f"New records appended to file: {rows_appended}")
    print(f"Total accumulated records in file: {total_rows}")
    print(f"Generated file: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
