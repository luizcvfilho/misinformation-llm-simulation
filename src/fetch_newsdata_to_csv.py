from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dotenv import load_dotenv
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen


API_URL = "https://newsdata.io/api/1/latest"
DEFAULT_OUTPUT = Path("data/newsdata_news.csv")
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


def save_csv(news: list[dict[str, Any]], output_path: Path) -> None:
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
	save_csv(news, args.output)

	print(f"Noticias salvas: {len(news)}")
	print(f"Arquivo gerado: {args.output}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
