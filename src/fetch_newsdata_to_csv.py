from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from dotenv import load_dotenv
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen


API_URL = "https://newsdata.io/api/1/latest"
DEFAULT_OUTPUT = Path("data/newsdata_news.csv")
QUERY_METADATA_ROW_ID = "__query_metadata__"
load_dotenv()

def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Busca noticias na API do NewsData.io e salva em CSV para uso no projeto."
		)
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=DEFAULT_OUTPUT,
		help="Caminho de saida do CSV (padrao: data/newsdata_news.csv).",
	)
	parser.add_argument(
		"--query",
		type=str,
		default="",
		help="Filtro de busca textual (parametro q da API).",
	)
	parser.add_argument(
		"--language",
		type=str,
		default="en",
		help="Idioma(s) separados por virgula (ex.: pt,en).",
	)
	parser.add_argument(
		"--country",
		type=str,
		default="",
		help="Pais(es) separados por virgula (ex.: br,us).",
	)
	parser.add_argument(
		"--category",
		type=str,
		default="",
		help="Categoria(s) separadas por virgula (ex.: politics,technology).",
	)
	parser.add_argument(
		"--max-records",
		type=int,
		default=200,
		help="Numero maximo de noticias para salvar.",
	)
	parser.add_argument(
		"--api-key-env",
		type=str,
		default="NEWSDATA_API_KEY",
		help="Nome da variavel de ambiente com a chave da API.",
	)
	return parser.parse_args()


def _request_news(params: dict[str, Any]) -> dict[str, Any]:
	query_string = urlencode(params)
	url = f"{API_URL}?{query_string}"

	try:
		with urlopen(url, timeout=30) as response:
			payload = response.read().decode("utf-8")
	except HTTPError as error:
		detail = error.read().decode("utf-8", errors="replace")
		raise RuntimeError(
			f"Erro HTTP {error.code} ao consultar NewsData.io: {detail}"
		) from error
	except URLError as error:
		raise RuntimeError(
			f"Falha de conexao ao consultar NewsData.io: {error.reason}"
		) from error

	try:
		return json.loads(payload)
	except json.JSONDecodeError as error:
		raise RuntimeError("Resposta da API nao esta em JSON valido.") from error


def _to_text(value: Any) -> str:
	if value is None:
		return ""
	if isinstance(value, list):
		return "; ".join(str(item) for item in value)
	return str(value)


def _split_multi_value(value: Any) -> list[str]:
	if value is None:
		return []

	if isinstance(value, list):
		items = value
	else:
		text = str(value)
		if not text.strip():
			return []
		# Alguns campos podem vir separados por ';' ou ','.
		items = []
		for chunk in text.split(";"):
			items.extend(chunk.split(","))

	cleaned = [str(item).strip() for item in items if str(item).strip()]
	return cleaned


def _count_values(news: list[dict[str, Any]], field: str) -> Counter[str]:
	counter: Counter[str] = Counter()
	for item in news:
		counter.update(_split_multi_value(item.get(field)))
	return counter


def _summarize_field(news: list[dict[str, Any]], field: str, top_n: int = 10) -> dict[str, Any]:
	counts = _count_values(news, field)
	return {
		"unique_count": len(counts),
		"top_values": [
			{"value": value, "count": count}
			for value, count in counts.most_common(top_n)
		],
	}


def _is_non_empty(value: Any) -> bool:
	if value is None:
		return False
	if isinstance(value, str):
		return bool(value.strip())
	if isinstance(value, list):
		return any(str(item).strip() for item in value)
	return True


def _summarize_query_results(news: list[dict[str, Any]], sample_size: int = 5) -> dict[str, Any]:
	pub_dates = [
		str(item.get("pubDate", "")).strip()
		for item in news
		if str(item.get("pubDate", "")).strip()
	]

	date_range = {
		"min_pubDate": min(pub_dates) if pub_dates else None,
		"max_pubDate": max(pub_dates) if pub_dates else None,
	}

	content_coverage = {
		"with_title": sum(1 for item in news if _is_non_empty(item.get("title"))),
		"with_description": sum(1 for item in news if _is_non_empty(item.get("description"))),
		"with_content": sum(1 for item in news if _is_non_empty(item.get("content"))),
		"with_full_description": sum(1 for item in news if _is_non_empty(item.get("full_description"))),
	}

	sample_articles = []
	for item in news[:sample_size]:
		sample_articles.append(
			{
				"title": _to_text(item.get("title")),
				"source_name": _to_text(item.get("source_name")),
				"pubDate": _to_text(item.get("pubDate")),
				"link": _to_text(item.get("link")),
			}
		)

	return {
		"rows_fetched": len(news),
		"date_range": date_range,
		"source_name_summary": _summarize_field(news, "source_name"),
		"language_summary": _summarize_field(news, "language"),
		"country_summary": _summarize_field(news, "country"),
		"category_summary": _summarize_field(news, "category"),
		"keyword_summary": _summarize_field(news, "keywords"),
		"content_coverage": content_coverage,
		"sample_articles": sample_articles,
	}


def fetch_news(
	api_key: str,
	query: str,
	language: str,
	country: str,
	category: str,
	max_records: int,
) -> list[dict[str, Any]]:
	records: list[dict[str, Any]] = []
	seen_ids: set[str] = set()
	next_page: str | None = None

	while len(records) < max_records:
		params: dict[str, Any] = {
			"apikey": api_key,
			"language": language,
		}
		if query:
			params["q"] = query
		if country:
			params["country"] = country
		if category:
			params["category"] = category
		if next_page:
			params["page"] = next_page

		data = _request_news(params)
		status = data.get("status")
		if status != "success":
			message = data.get("results", data)
			raise RuntimeError(f"API retornou status inesperado: {status} | {message}")

		batch = data.get("results", [])
		if not isinstance(batch, list) or not batch:
			break

		for item in batch:
			article_id = _to_text(item.get("article_id")).strip()
			fallback_key = _to_text(item.get("link")).strip()
			unique_key = article_id or fallback_key
			if unique_key and unique_key in seen_ids:
				continue
			if unique_key:
				seen_ids.add(unique_key)

			records.append(item)
			if len(records) >= max_records:
				break

		next_page = data.get("nextPage")
		if not next_page:
			break

	return records


def _build_query_metadata(
	*,
	query: str,
	language: str,
	country: str,
	category: str,
	max_records: int,
	news: list[dict[str, Any]],
) -> dict[str, Any]:
	return {
		"query_parameters": {
			"query": query,
			"language": language,
			"country": country,
			"category": category,
			"max_records": max_records,
		},
		"query_results_summary": _summarize_query_results(news),
		"fetched_at_utc": datetime.now(timezone.utc).isoformat(),
	}


def save_csv(
	news: list[dict[str, Any]],
	output_path: Path,
	query_metadata: dict[str, Any],
) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)

	columns = [
		"article_id",
		"title",
		"link",
		"description",
		"content",
		"full_description",
		"pubDate",
		"pubDateTZ",
		"image_url",
		"video_url",
		"source_id",
		"source_name",
		"source_priority",
		"source_url",
		"source_icon",
		"language",
		"country",
		"category",
		"creator",
		"keywords",
		"duplicate",
	]

	with output_path.open("w", encoding="utf-8", newline="") as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=columns)
		writer.writeheader()

		metadata_row = {column: "" for column in columns}
		metadata_row["article_id"] = QUERY_METADATA_ROW_ID
		metadata_row["title"] = "QUERY_METADATA"
		metadata_row["description"] = json.dumps(query_metadata, ensure_ascii=False)
		metadata_row["language"] = _to_text(query_metadata.get("language"))
		metadata_row["country"] = _to_text(query_metadata.get("country"))
		metadata_row["category"] = _to_text(query_metadata.get("category"))
		writer.writerow(metadata_row)

		for item in news:
			row = {column: _to_text(item.get(column)) for column in columns}
			writer.writerow(row)


def main() -> int:
	args = parse_args()
	if args.max_records <= 0:
		print("Erro: --max-records deve ser maior que zero.", file=sys.stderr)
		return 2

	api_key = os.getenv(args.api_key_env, "").strip()
	if not api_key:
		print(
			(
				f"Erro: variavel de ambiente {args.api_key_env} nao encontrada. "
				"Defina sua chave da NewsData.io antes de executar."
			),
			file=sys.stderr,
		)
		return 2

	news = fetch_news(
		api_key=api_key,
		query=args.query.strip(),
		language=args.language.strip(),
		country=args.country.strip(),
		category=args.category.strip(),
		max_records=args.max_records,
	)
	query_metadata = _build_query_metadata(
		query=args.query.strip(),
		language=args.language.strip(),
		country=args.country.strip(),
		category=args.category.strip(),
		max_records=args.max_records,
		news=news,
	)
	save_csv(news, args.output, query_metadata)

	print(f"Noticias salvas: {len(news)}")
	print(f"Arquivo gerado: {args.output}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
