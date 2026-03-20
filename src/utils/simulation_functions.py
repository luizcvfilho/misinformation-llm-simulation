from __future__ import annotations

import os
import random
import time
from collections.abc import Callable

import pandas as pd
from enums.providers import Provider
from google import genai
from google.genai import types
from openai import OpenAI

TEXT_COLUMN_CANDIDATES = ("full_description", "content", "description", "title")
DETAILED_TEXT_COLUMNS = ("full_description", "content", "description")

LANGUAGE_CODE_TO_NAME = {
	"en": "English",
	"pt": "Portuguese",
	"es": "Spanish",
	"fr": "French",
	"de": "German",
	"it": "Italian",
}


def choose_news_text_column(
	df: pd.DataFrame,
	candidates=TEXT_COLUMN_CANDIDATES,
	prefer_detailed_text: bool = True,
) -> str:
	if prefer_detailed_text:
		priority_candidates = [column for column in DETAILED_TEXT_COLUMNS if column in candidates]
	else:
		priority_candidates = list(candidates)

	fallback_candidates = [column for column in candidates if column not in priority_candidates]
	ordered_candidates = [*priority_candidates, *fallback_candidates]

	best_column = None
	best_total_chars = -1

	for column in ordered_candidates:
		if column not in df.columns:
			continue

		series = df[column].fillna("").astype(str).str.strip()
		non_empty_series = series[series.ne("")]
		if non_empty_series.empty:
			continue

		total_chars = int(non_empty_series.str.len().sum())
		if total_chars > best_total_chars:
			best_total_chars = total_chars
			best_column = column

	if best_column is not None:
		return best_column

	raise ValueError("Nenhuma coluna de texto foi encontrada. Informe 'text_column' manualmente.")


def resolve_row_text(
	row: pd.Series,
	preferred_column: str | None = None,
	candidates=TEXT_COLUMN_CANDIDATES,
	allow_title_fallback: bool = False,
) -> tuple[str, str]:
	ordered_candidates = []
	if preferred_column:
		ordered_candidates.append(preferred_column)

	ordered_candidates.extend(column for column in candidates if column not in ordered_candidates)

	if not allow_title_fallback:
		ordered_candidates = [column for column in ordered_candidates if column != "title"]

	for column in ordered_candidates:
		if column not in row.index:
			continue

		value = row[column]
		text = "" if pd.isna(value) else str(value).strip()
		if text:
			return column, text

	if allow_title_fallback and "title" in row.index:
		title_value = row["title"]
		title_text = "" if pd.isna(title_value) else str(title_value).strip()
		if title_text:
			return "title", title_text

	raise ValueError("A linha nao possui texto utilizavel nas colunas candidatas.")


def normalize_language_code(raw_language: str | None) -> str | None:
	if not raw_language:
		return None

	value = raw_language.strip().lower()
	if not value:
		return None

	aliases = {
		"english": "en",
		"ingles": "en",
		"portuguese": "pt",
		"portugues": "pt",
		"spanish": "es",
		"espanol": "es",
		"espanhol": "es",
		"french": "fr",
		"german": "de",
		"italian": "it",
	}

	if value in aliases:
		return aliases[value]

	if "," in value:
		first = value.split(",", maxsplit=1)[0].strip()
		if first in aliases:
			return aliases[first]
		if len(first) == 2 and first.isalpha():
			return first

	if len(value) == 2 and value.isalpha():
		return value

	return None


def infer_text_language(text: str) -> str | None:
	sample = f" {text.lower()} "

	english_markers = [
		" the ",
		" and ",
		" of ",
		" to ",
		" in ",
		" is ",
		" for ",
		" on ",
		" with ",
		" that ",
	]
	portuguese_markers = [
		" que ",
		" de ",
		" para ",
		" com ",
		" uma ",
		" nao ",
		" na ",
		" no ",
		" os ",
		" as ",
		" em ",
	]

	en_score = sum(sample.count(marker) for marker in english_markers)
	pt_score = sum(sample.count(marker) for marker in portuguese_markers)

	if pt_score >= 2 and pt_score > en_score:
		return "pt"
	if en_score >= 2 and en_score >= pt_score:
		return "en"
	return None


def resolve_output_language(row: pd.Series, original_text: str) -> tuple[str, str]:
	if "language" in row.index and pd.notna(row["language"]):
		row_language = normalize_language_code(str(row["language"]))
		if row_language:
			return row_language, "row.language"

	if "country" in row.index and pd.notna(row["country"]):
		countries = str(row["country"]).lower()
		if "br" in countries:
			return "pt", "row.country"

	inferred = infer_text_language(original_text)
	if inferred:
		return inferred, "heuristic"

	return "en", "default"


def generate_rewrite_with_retry_gemini(
	client: genai.Client,
	*,
	model: str,
	prompt: str,
	system_instruction: str,
	max_attempts: int = 5,
	base_delay: float = 2.0,
	before_request_hook: Callable[[], None] | None = None,
) -> str:
	last_error = None

	for attempt in range(1, max_attempts + 1):
		try:
			if before_request_hook is not None:
				before_request_hook()

			response = client.models.generate_content(
				model=model,
				config=types.GenerateContentConfig(
					system_instruction=system_instruction,
					temperature=0.8,
				),
				contents=prompt,
			)

			rewritten_text = (response.text or "").strip()
			if not rewritten_text:
				raise ValueError("A resposta da API veio vazia.")

			return rewritten_text
		except Exception as exc:
			last_error = exc
			error_text = str(exc)
			is_retryable = (
				"429" in error_text
				or "RESOURCE_EXHAUSTED" in error_text
				or "503" in error_text
				or "UNAVAILABLE" in error_text
			)

			if not is_retryable or attempt == max_attempts:
				break

			sleep_for = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
			time.sleep(sleep_for)

	raise last_error


def generate_rewrite_with_retry_openai_compatible(
	client: OpenAI,
	*,
	model: str,
	prompt: str,
	system_instruction: str,
	max_attempts: int = 5,
	base_delay: float = 2.0,
	before_request_hook: Callable[[], None] | None = None,
) -> str:
	last_error = None

	for attempt in range(1, max_attempts + 1):
		try:
			if before_request_hook is not None:
				before_request_hook()

			response = client.chat.completions.create(
				model=model,
				temperature=0.8,
				messages=[
					{"role": "system", "content": system_instruction},
					{"role": "user", "content": prompt},
				],
			)

			rewritten_text = (response.choices[0].message.content or "").strip()
			if not rewritten_text:
				raise ValueError("A resposta da API veio vazia.")

			return rewritten_text
		except Exception as exc:
			last_error = exc
			error_text = str(exc)
			is_retryable = (
				"429" in error_text
				or "503" in error_text
				or "UNAVAILABLE" in error_text
				or "rate limit" in error_text.lower()
				or "timeout" in error_text.lower()
			)

			if not is_retryable or attempt == max_attempts:
				break

			sleep_for = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
			time.sleep(sleep_for)

	raise last_error


def rewrite_news_with_personality(
	df: pd.DataFrame,
	personality: str,
	text_column: str | None = None,
	title_column: str = "title",
	model: str = "gemini-2.5-flash-lite",
	provider: Provider | str = Provider.GEMINI,
	api_key: str | None = None,
	base_url: str | None = None,
	output_column: str = "rewritten_news",
	max_rows: int | None = None,
	sleep_seconds: float = 0.0,
	max_requests_per_minute: int | None = None,
	retry_attempts: int = 5,
	allow_title_fallback: bool = False,
) -> pd.DataFrame:
	if df.empty:
		raise ValueError("O DataFrame esta vazio.")

	if not personality or not personality.strip():
		raise ValueError("Informe uma personalidade valida.")

	if max_requests_per_minute is not None and max_requests_per_minute <= 0:
		raise ValueError("'max_requests_per_minute' deve ser maior que zero quando informado.")

	provider_normalized = provider.value if isinstance(provider, Provider) else str(provider).strip().lower()
	if provider_normalized not in {item.value for item in Provider}:
		raise ValueError("Provider invalido. Use: gemini, openrouter, deepseek ou local.")

	env_key_name = None
	if provider_normalized == "gemini":
		env_key_name = "GEMINI_API_KEY"
	elif provider_normalized == "openrouter":
		env_key_name = "OPENROUTER_API_KEY"
	elif provider_normalized == "deepseek":
		env_key_name = "DEEPSEEK_API_KEY"

	if provider_normalized == "local":
		resolved_api_key = api_key or os.getenv("LOCAL_OPENAI_API_KEY") or "ollama"
		resolved_base_url = base_url or os.getenv("LOCAL_OPENAI_BASE_URL") or "http://127.0.0.1:11434/v1"
	else:
		resolved_api_key = api_key or os.getenv(env_key_name)
		if not resolved_api_key:
			raise ValueError(f"Defina a {env_key_name} no ambiente ou passe a chave em 'api_key'.")

	resolved_text_column = text_column or choose_news_text_column(df)
	if resolved_text_column not in df.columns:
		raise ValueError(f"A coluna '{resolved_text_column}' nao existe no DataFrame.")

	if provider_normalized == "gemini":
		client = genai.Client(api_key=resolved_api_key)
	elif provider_normalized == "openrouter":
		client = OpenAI(
			api_key=resolved_api_key,
			base_url="https://openrouter.ai/api/v1",
			default_headers={
				"HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", ""),
				"X-Title": os.getenv("OPENROUTER_X_TITLE", "misinformation-llm-simulation"),
			},
		)
	elif provider_normalized == "deepseek":
		client = OpenAI(
			api_key=resolved_api_key,
			base_url="https://api.deepseek.com",
		)
	else:
		client = OpenAI(
			api_key=resolved_api_key,
			base_url=resolved_base_url,
		)

	rewritten_df = df.copy()
	rewritten_df["source_text_column"] = pd.NA
	rewritten_df["target_language"] = pd.NA
	rewritten_df["target_language_source"] = pd.NA
	rewritten_df[output_column] = pd.NA
	rewritten_df["rewrite_status"] = "not_requested"
	rewritten_df["rewrite_error"] = pd.NA

	target_indexes = list(rewritten_df.index)
	if max_rows is not None:
		target_indexes = target_indexes[:max_rows]

	system_instruction = (
		"You are a news rewriting assistant. Keep the requested output language exactly, "
		"preserve the central facts, and do not add new information."
	)

	current_minute_bucket = int(time.time() // 60)
	requests_in_current_minute = 0

	def wait_for_rate_limit_slot() -> None:
		nonlocal current_minute_bucket, requests_in_current_minute

		if max_requests_per_minute is None:
			return

		now_bucket = int(time.time() // 60)
		if now_bucket != current_minute_bucket:
			current_minute_bucket = now_bucket
			requests_in_current_minute = 0

		if requests_in_current_minute >= max_requests_per_minute:
			seconds_until_next_minute = 60 - (time.time() % 60)
			time.sleep(seconds_until_next_minute)
			current_minute_bucket = int(time.time() // 60)
			requests_in_current_minute = 0

		requests_in_current_minute += 1

	for row_index in target_indexes:
		row = rewritten_df.loc[row_index]

		try:
			source_column, original_text = resolve_row_text(
				row=row,
				preferred_column=resolved_text_column,
				allow_title_fallback=allow_title_fallback,
			)
			rewritten_df.at[row_index, "source_text_column"] = source_column
		except ValueError as exc:
			rewritten_df.at[row_index, "rewrite_status"] = "skipped"
			rewritten_df.at[row_index, "rewrite_error"] = str(exc)
			continue

		target_language_code, target_language_source = resolve_output_language(
			row=row,
			original_text=original_text,
		)
		target_language_name = LANGUAGE_CODE_TO_NAME.get(
			target_language_code,
			target_language_code.upper(),
		)
		rewritten_df.at[row_index, "target_language"] = target_language_code
		rewritten_df.at[row_index, "target_language_source"] = target_language_source

		title = ""
		if title_column in rewritten_df.columns and pd.notna(rewritten_df.at[row_index, title_column]):
			title = str(rewritten_df.at[row_index, title_column]).strip()

		prompt = f"""
				Adopt the following personality while rewriting the news article:
				{personality}

				Required output language: {target_language_name} ({target_language_code}).

				Rules:
				- Write strictly in {target_language_name} ({target_language_code}).
				- Do not translate into another language different from the requested one.
				- Preserve the central facts.
				- Do not invent data, numbers, quotes, or characters.
				- Adjust tone, vocabulary, and style to reflect the requested personality.
				- Return only the rewritten text.

				Title:
				{title or 'Untitled'}

				Original text:
				{original_text}
				""".strip()

		try:
			if provider_normalized == "gemini":
				rewritten_text = generate_rewrite_with_retry_gemini(
					client,
					model=model,
					prompt=prompt,
					system_instruction=system_instruction,
					max_attempts=retry_attempts,
					before_request_hook=wait_for_rate_limit_slot,
				)
			else:
				rewritten_text = generate_rewrite_with_retry_openai_compatible(
					client,
					model=model,
					prompt=prompt,
					system_instruction=system_instruction,
					max_attempts=retry_attempts,
					before_request_hook=wait_for_rate_limit_slot,
				)

			rewritten_df.at[row_index, output_column] = rewritten_text
			rewritten_df.at[row_index, "rewrite_status"] = "success"
			rewritten_df.at[row_index, "rewrite_error"] = pd.NA
		except Exception as exc:
			rewritten_df.at[row_index, "rewrite_status"] = "error"
			rewritten_df.at[row_index, "rewrite_error"] = str(exc)

		if sleep_seconds > 0:
			time.sleep(sleep_seconds)

	return rewritten_df
