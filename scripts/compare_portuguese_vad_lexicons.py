from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import sys
import unicodedata
import zipfile
from collections import defaultdict
from datetime import UTC, datetime
from importlib.metadata import version
from itertools import combinations
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from misinformation_simulation.text_metrics.nrc_vad import TOKEN_PATTERN  # noqa: E402
from misinformation_simulation.text_metrics.vad import (  # noqa: E402
    DEFAULT_VAD_MODEL_NAME,
    load_huggingface_vad_model,
    predict_vad_batch,
)

CACHE_DIR = PROJECT_ROOT / ".cache/vad_lexicon_comparison"
OUTPUT_DIR = PROJECT_ROOT / "output/audit/PortugueseVADLexiconComparison"
INPUT_PATH = PROJECT_ROOT / "output/elections_2026/simulation_candidates.csv"
SELECTED_POSTS = {
    "26242664535431023": "Flood rescue",
    "1094879369551356": "Stadium accident",
    "1681964447272553": "Hailstorm",
}
TSE_URL = (
    "https://www.tse.jus.br/comunicacao/noticias/2026/Setembro/"
    "mesarios-tem-ate-3-de-outubro-para-concluir-treinamento-das-eleicoes-2026"
)
TSE_TITLE = "Mesários têm até 3 de outubro para concluir treinamento das Eleições 2026"
HAILSTORM_TRANSLATION = """Storm - A heavy hailstorm hit Mafra, in northern Santa Catarina,
on Tuesday afternoon (the 1st). Hailstones accumulated and the streets turned white.

Santa Catarina has recently experienced hailstorms. Civil Defense reported that nine
municipalities had declared a state of emergency since Friday (the 28th) because of
damage caused by hailstones and accumulated rain.

Teams are on site, and there was no information about damage when this report was
published. According to Civil Defense, Wednesday (the 2nd) and Thursday (the 3rd)
should have stable weather in the state.

Learn more on #g1.

#granizo #temporal #g1santacatarina #notícias"""
DIMENSIONS = ("valence", "arousal", "dominance")
NRC_MEMBER_PT = "NRC-VAD-Lexicon/OneFilePerLanguage/Portuguese-NRC-VAD-Lexicon.txt"
NRC_MEMBER_EN = "NRC-VAD-Lexicon/NRC-VAD-Lexicon.txt"
NRC_V2_PATH = PROJECT_ROOT / ".cache/nrc_vad/NRC-VAD-Lexicon-v2.1.zip"
NRC_V2_MEMBER = "NRC-VAD-Lexicon-v2.1/NRC-VAD-Lexicon-v2.1.txt"
MODELS = ("nrc_v1", "memolon", "nrc_v2_1", "current_model")
LEXICON_MODELS = MODELS[:-1]
MODEL_LABELS = {
    "nrc_v1": "NRC v1 (Portuguese translation / English original)",
    "memolon": "MEmoLon MTL_grouped",
    "nrc_v2_1": "NRC v2.1 (English lexicon)",
    "current_model": DEFAULT_VAD_MODEL_NAME,
}
NEWS_TRANSLATIONS = {
    "26242664535431023": """🌊 EMOTIONAL RESCUE | A 2-year-old child was rescued alive by
soldiers of the Nepal Army amid the wreckage left by severe floods that hit the
Nepal-Tibet border region on Wednesday (the 26th).

The child was found among the rubble in one of the areas hit by flash floods.
Military personnel managed to remove the child from the site and take the child
to a safe area. Images show teams carrying children and helping residents during
the rescues.

More than 5,000 Army and police personnel are taking part in search operations
and assistance to the victims. The teams continue working to locate other people
and assist the affected communities.

📲 Learn more at meionews.com

Nepal | Floods | Rescue | Army | Tibet | Tragedy

Credits: Clip Lions | @myhoodbr""",
    "1094879369551356": """ACCIDENT! 😱 A Coritiba fan fell from the second tier of the
stands at Couto Pereira during the derby against Athletico, played on Friday
night (the 11th). The serious accident occurred inside the stadium while the
Parana derby was underway.

The incident was confirmed by the Mobile Police Unit for Football and Events
(Demafe). According to the security authorities, the fan was promptly assisted
by the emergency teams at the site and taken by ambulance to Hospital do
Trabalhador (HT), in the capital of Parana.

🔗 Read the full article at bandab.com.br

📹 Social Media

#acidente #coutopereira #queda #atletiba #bandab""",
    "1681964447272553": HAILSTORM_TRANSLATION,
}


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold().replace("’", "'")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_samples() -> list[dict]:
    with INPUT_PATH.open(encoding="utf-8-sig", newline="") as handle:
        posts = {row["id"]: row for row in csv.DictReader(handle)}
    samples = []
    for post_id, label in SELECTED_POSTS.items():
        post = posts[post_id]
        if post["lang"] != "pt":
            raise ValueError(f"Expected Portuguese post: {post_id}")
        samples.append(
            dict(
                sample_id=post_id,
                label=label,
                kind="local_news_post",
                language="pt",
                text=post["text"],
                source_url=post["mcl_url"],
                published_at=post["creation_time"],
            )
        )
        samples.append(
            dict(
                sample_id=f"{post_id}_en",
                label=f"{label} (translation)",
                kind="manual_translation_control",
                language="en",
                text=NEWS_TRANSLATIONS[post_id],
                translation_of=post_id,
            )
        )
    samples.append(
        dict(
            sample_id="tse_training_title",
            label="Poll worker training headline",
            kind="coverage_only_title",
            language="pt",
            text=TSE_TITLE,
            source_url=TSE_URL,
        )
    )
    word_controls = {
        "pt": {
            "valence": ("feliz contente satisfeito", "triste infeliz assustado"),
            "arousal": ("agitado excitado alerta", "tranquilo relaxado sonolento"),
            "dominance": ("poderoso confiante dominante", "fraco indefeso vulnerável"),
        },
        "en": {
            "valence": ("happy joyful delighted", "sad miserable terrified"),
            "arousal": ("agitated excited alert", "calm relaxed sleepy"),
            "dominance": ("powerful confident dominant", "weak helpless vulnerable"),
        },
    }
    negation_controls = {
        "pt": {
            "valence": ("Estou feliz.", "Não estou feliz."),
            "arousal": ("Estou agitado.", "Não estou agitado."),
            "dominance": ("Tenho controle da situação.", "Não tenho controle da situação."),
        },
        "en": {
            "valence": ("I am happy.", "I am not happy."),
            "arousal": ("I am agitated.", "I am not agitated."),
            "dominance": (
                "I have control of the situation.",
                "I do not have control of the situation.",
            ),
        },
    }
    scope_controls = {
        "pt": {
            "valence": (
                "Estou feliz, mas não estou triste.",
                "Estou triste, mas não estou feliz.",
            )
        },
        "en": {
            "valence": (
                "I am happy, but I am not sad.",
                "I am sad, but I am not happy.",
            )
        },
    }
    for kind, controls, directions in (
        ("affect_word_control", word_controls, ("high", "low")),
        ("negation_control", negation_controls, ("affirmative", "negated")),
        ("negation_scope_control", scope_controls, ("high", "low")),
    ):
        for language, dimensions in controls.items():
            for dimension, texts in dimensions.items():
                for direction, text in zip(directions, texts, strict=True):
                    sample_id = f"{kind}_{language}_{dimension}_{direction}"
                    samples.append(
                        dict(
                            sample_id=sample_id,
                            label=sample_id,
                            kind=kind,
                            language=language,
                            text=text,
                            dimension=dimension,
                            direction=direction,
                        )
                    )
    return samples


def candidate_terms(samples: list[dict], language: str) -> set[str]:
    candidates = set()
    for sample in samples:
        if sample["language"] != language:
            continue
        text = normalize(sample["text"])
        tokens = list(TOKEN_PATTERN.finditer(text))
        for start, token in enumerate(tokens):
            term = token.group()
            candidates.add(term)
            for end in range(start + 1, len(tokens)):
                if not text[tokens[end - 1].end() : tokens[end].start()].isspace():
                    break
                term += " " + tokens[end].group()
                candidates.add(term)
    return candidates


def load_lexicon(model: str, language: str, candidates: set[str]) -> tuple[dict, dict]:
    retained = defaultdict(list)
    row_count = out_of_range = 0
    minimum, maximum = [math.inf] * 3, [-math.inf] * 3
    translated_terms = set()
    duplicate_count = unsupported_count = 0
    if model == "nrc_v1":
        member = NRC_MEMBER_PT if language == "pt" else NRC_MEMBER_EN
        with zipfile.ZipFile(CACHE_DIR / "NRC-VAD-Lexicon-v1.zip") as archive:
            handle = io.StringIO(archive.read(member).decode("utf-8-sig"))
        reader = csv.reader(handle, delimiter="\t")
        if language == "pt":
            header = next(reader)
            if header != ["English Word", "Valence", "Arousal", "Dominance", "Portuguese Word"]:
                raise ValueError(f"Unexpected NRC Portuguese header: {header}")
        bounds = (0, 1)
    elif model == "nrc_v2_1":
        member = NRC_V2_MEMBER
        with zipfile.ZipFile(NRC_V2_PATH) as archive:
            handle = io.StringIO(archive.read(member).decode("utf-8-sig"))
        reader = csv.reader(handle, delimiter="\t")
        if next(reader) != ["term", *DIMENSIONS]:
            raise ValueError("Unexpected NRC v2.1 header")
        bounds = (-1, 1)
    else:
        member = f"{language}.tsv"
        handle = (CACHE_DIR / f"memolon_{language}.tsv").open(encoding="utf-8", newline="")
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        if header[:4] != ["word", *DIMENSIONS]:
            raise ValueError(f"Unexpected MEmoLon header: {header}")
        bounds = (1, 9)
    with handle:
        for row in reader:
            row_count += 1
            term = normalize(row[4] if model == "nrc_v1" and language == "pt" else row[0]).strip()
            tokens = term.split()
            if not tokens or any(TOKEN_PATTERN.fullmatch(token) is None for token in tokens):
                unsupported_count += 1
                continue
            term = " ".join(tokens)
            values = tuple(float(value) for value in row[1:4])
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Nonfinite values: {model} {language} {term}")
            for index, value in enumerate(values):
                minimum[index] = min(minimum[index], value)
                maximum[index] = max(maximum[index], value)
            out_of_range += any(not bounds[0] <= value <= bounds[1] for value in values)
            if model.startswith("nrc_"):
                duplicate_count += term in translated_terms
                translated_terms.add(term)
            if term in candidates:
                retained[term].append(values)
    entries = {
        term: dict(
            native=tuple(math.fsum(v[i] for v in values) / len(values) for i in range(3)),
            source_rows=len(values),
        )
        for term, values in retained.items()
    }
    metadata = dict(
        model=model,
        language=language,
        member=member,
        source_rows=row_count,
        unsupported_rows=unsupported_count,
        duplicate_supported_nrc_rows=duplicate_count,
        unique_supported_nrc_terms=len(translated_terms) if model.startswith("nrc_") else None,
        retained_input_relevant_terms=len(entries),
        retained_source_rows=sum(len(values) for values in retained.values()),
        retained_duplicate_rows=sum(len(values) - 1 for values in retained.values()),
        retained_out_of_range_source_rows=sum(
            any(not bounds[0] <= value <= bounds[1] for value in values)
            for values_list in retained.values()
            for values in values_list
        ),
        native_bounds=bounds,
        observed_min=minimum,
        observed_max=maximum,
        out_of_range_supported_rows=out_of_range,
        resource_language="en" if model == "nrc_v2_1" else language,
        portuguese_validation="Unsupported-language probe"
        if model == "nrc_v2_1"
        else "Lexical resource available",
    )
    return entries, metadata


def score_sample(sample: dict, model: str, entries: dict) -> tuple[dict, list[dict]]:
    trie = {}
    for term, entry in entries.items():
        node = trie
        for token in term.split():
            node = node.setdefault(token, {})
        node[None] = (term, entry)
    text = normalize(sample["text"])
    tokens = list(TOKEN_PATTERN.finditer(text))
    matches = []
    matched_indices = set()
    index = 0
    while index < len(tokens):
        node, cursor, best = trie, index, None
        while cursor < len(tokens):
            if (
                cursor > index
                and not text[tokens[cursor - 1].end() : tokens[cursor].start()].isspace()
            ):
                break
            node = node.get(tokens[cursor].group())
            if node is None:
                break
            cursor += 1
            if None in node:
                best = (cursor, *node[None])
        if best is None:
            index += 1
            continue
        end, term, entry = best
        scaled = tuple(
            1 + 4 * value
            if model == "nrc_v1"
            else 3 + 2 * value
            if model == "nrc_v2_1"
            else (value + 1) / 2
            for value in entry["native"]
        )
        matches.append(
            dict(
                sample_id=sample["sample_id"],
                model=model,
                language=sample["language"],
                term=term,
                token_start=index,
                token_end=end,
                source_rows=entry["source_rows"],
                **dict(zip(DIMENSIONS, scaled, strict=True)),
            )
        )
        matched_indices.update(range(index, end))
        index = end
    row = {key: sample[key] for key in ("sample_id", "label", "kind", "language")}
    row.update(
        model=model,
        token_count=len(tokens),
        matched_token_count=len(matched_indices),
        matched_term_count=len(matches),
        multiword_match_count=sum(" " in m["term"] for m in matches),
        token_coverage=len(matched_indices) / len(tokens) if tokens else 0,
        unmatched_tokens=json.dumps(
            [t.group() for i, t in enumerate(tokens) if i not in matched_indices],
            ensure_ascii=False,
        ),
        model_type="lexicon",
        evaluation_status=(
            "unsupported_language_probe"
            if model == "nrc_v2_1" and sample["language"] == "pt"
            else "language_resource_available"
        ),
        wordpiece_count=None,
        unknown_wordpiece_count=None,
        input_truncated=None,
    )
    for dimension in DIMENSIONS:
        row[dimension] = (
            math.fsum(match[dimension] for match in matches) / len(matches)
            if matches and sample["kind"] != "coverage_only_title"
            else None
        )
    return row, matches


def score_current_model(samples: list[dict]) -> tuple[list[dict], dict]:
    snapshot = Path(snapshot_download(DEFAULT_VAD_MODEL_NAME, local_files_only=True))
    torch.set_num_threads(4)
    bundle = load_huggingface_vad_model(str(snapshot), device="cpu", max_length=512)
    inputs = [sample for sample in samples if sample["kind"] != "coverage_only_title"]
    scores = predict_vad_batch(
        [sample["text"] for sample in inputs], model_bundle=bundle, batch_size=8
    )
    scores_by_id = {
        sample["sample_id"]: score for sample, score in zip(inputs, scores, strict=True)
    }
    rows = []
    for sample in samples:
        row = {key: sample[key] for key in ("sample_id", "label", "kind", "language")}
        wordpieces = bundle.tokenizer(sample["text"], add_special_tokens=True, truncation=False)[
            "input_ids"
        ]
        row.update(
            model="current_model",
            token_count=len(TOKEN_PATTERN.findall(normalize(sample["text"]))),
            matched_token_count=None,
            matched_term_count=None,
            multiword_match_count=None,
            token_coverage=None,
            unmatched_tokens=None,
            model_type="contextual_regression",
            evaluation_status=(
                "unsupported_language_probe"
                if sample["language"] == "pt"
                else "documented_training_language"
            ),
            wordpiece_count=len(wordpieces),
            unknown_wordpiece_count=wordpieces.count(bundle.tokenizer.unk_token_id),
            input_truncated=len(wordpieces) > bundle.max_length,
        )
        score = scores_by_id.get(sample["sample_id"])
        for dimension in DIMENSIONS:
            row[dimension] = getattr(score, dimension) if score is not None else None
        rows.append(row)
    metadata = dict(
        model="current_model",
        model_id=DEFAULT_VAD_MODEL_NAME,
        revision=snapshot.name,
        cached_snapshot=str(snapshot),
        device=bundle.device,
        max_length=bundle.max_length,
        batch_size=8,
        torch_threads=torch.get_num_threads(),
        torch_version=version("torch"),
        transformers_version=version("transformers"),
        inference=(
            "Unmodified project load_huggingface_vad_model and predict_vad_batch; "
            "eval and inference_mode."
        ),
        weights_sha256={
            path.name: sha256(path) for path in sorted(snapshot.iterdir()) if path.is_file()
        },
        native_output=(
            "Raw regression logits, used as nominal 1-5 VAD by the project; "
            "no clipping or rescaling."
        ),
        resource_language="en",
        portuguese_validation="No Portuguese fine-tuning or validation documented on model card.",
    )
    return rows, metadata


def write_csv(name: str, rows: list[dict]) -> None:
    with (OUTPUT_DIR / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    samples = build_samples()
    download_manifest = CACHE_DIR / "pilot_download_manifest.json"
    if not download_manifest.exists():
        download_manifest = CACHE_DIR / "download_manifest.json"
    downloads = json.loads(download_manifest.read_text(encoding="utf-8"))
    nrc_path = CACHE_DIR / "NRC-VAD-Lexicon-v1.zip"
    if sha256(nrc_path) != downloads["nrc"]["sha256"]:
        raise ValueError("NRC resource hash changed")
    for member in downloads["memolon"]["members"]:
        if sha256(CACHE_DIR / f"memolon_{member['language']}.tsv") != member["sha256"]:
            raise ValueError("MEmoLon resource hash changed")
    rows, matches, resources = [], [], []
    for language in ("pt", "en"):
        candidates = candidate_terms(samples, language)
        for model in LEXICON_MODELS:
            entries, metadata = load_lexicon(model, language, candidates)
            resources.append(metadata)
            for sample in samples:
                if sample["language"] == language:
                    row, trace = score_sample(sample, model, entries)
                    rows.append(row)
                    matches.extend(trace)
    print("Running the current project model on the cached snapshot.", flush=True)
    neural_rows, neural_metadata = score_current_model(samples)
    rows.extend(neural_rows)
    resources.append(neural_metadata)
    by_key = {(row["sample_id"], row["model"]): row for row in rows}
    summaries = []
    for language in ("pt", "en"):
        for model in MODELS:
            group = [
                row
                for row in rows
                if row["language"] == language
                and row["model"] == model
                and row["kind"] in ("local_news_post", "manual_translation_control")
            ]
            summaries.append(
                dict(
                    language=language,
                    model=model,
                    news_count=len(group),
                    token_count=sum(row["token_count"] for row in group),
                    matched_tokens=(
                        sum(row["matched_token_count"] for row in group)
                        if model != "current_model"
                        else None
                    ),
                    weighted_coverage=(
                        sum(row["matched_token_count"] for row in group)
                        / sum(row["token_count"] for row in group)
                        if model != "current_model"
                        else None
                    ),
                )
            )
    comparisons = []
    for sample in samples:
        for model_a, model_b in combinations(MODELS, 2):
            first, second = (by_key[(sample["sample_id"], model)] for model in (model_a, model_b))
            comparison = dict(
                sample_id=sample["sample_id"],
                language=sample["language"],
                kind=sample["kind"],
                model_a=model_a,
                model_b=model_b,
                coverage_delta_b_minus_a=(
                    second["token_coverage"] - first["token_coverage"]
                    if first["token_coverage"] is not None and second["token_coverage"] is not None
                    else None
                ),
            )
            for dimension in DIMENSIONS:
                comparison[f"{dimension}_delta_b_minus_a"] = (
                    second[dimension] - first[dimension]
                    if second[dimension] is not None and first[dimension] is not None
                    else None
                )
            comparisons.append(comparison)
    controls = []
    for kind, high, low in (
        ("affect_word_control", "high", "low"),
        ("negation_control", "affirmative", "negated"),
        ("negation_scope_control", "high", "low"),
    ):
        for language in ("pt", "en"):
            for dimension in ("valence",) if kind == "negation_scope_control" else DIMENSIONS:
                for model in MODELS:
                    high_row = by_key[(f"{kind}_{language}_{dimension}_{high}", model)]
                    low_row = by_key[(f"{kind}_{language}_{dimension}_{low}", model)]
                    delta = (
                        high_row[dimension] - low_row[dimension]
                        if high_row[dimension] is not None and low_row[dimension] is not None
                        else None
                    )
                    controls.append(
                        dict(
                            kind=kind,
                            language=language,
                            dimension=dimension,
                            model=model,
                            higher_condition_score=high_row[dimension],
                            lower_condition_score=low_row[dimension],
                            expected_direction_delta=delta,
                            expected_direction_observed=delta > 0 if delta is not None else None,
                        )
                    )
    translation_deltas = []
    for post_id in SELECTED_POSTS:
        for model in MODELS:
            pt, en = (by_key[(sample_id, model)] for sample_id in (post_id, f"{post_id}_en"))
            translation_deltas.append(
                dict(
                    sample_id=post_id,
                    model=model,
                    **{
                        f"{dim}_en_minus_pt": en[dim] - pt[dim]
                        if en[dim] is not None and pt[dim] is not None
                        else None
                        for dim in DIMENSIONS
                    },
                )
            )
    control_summaries = []
    for model in MODELS:
        for language in ("pt", "en"):
            for kind in ("affect_word_control", "negation_control", "negation_scope_control"):
                group = [
                    row
                    for row in controls
                    if row["model"] == model and row["language"] == language and row["kind"] == kind
                ]
                control_summaries.append(
                    dict(
                        model=model,
                        language=language,
                        kind=kind,
                        contrasts=len(group),
                        available=sum(row["expected_direction_delta"] is not None for row in group),
                        expected_direction_observed=sum(
                            row["expected_direction_observed"] is True for row in group
                        ),
                        missing=sum(row["expected_direction_delta"] is None for row in group),
                    )
                )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, values in (
        ("scores.csv", rows),
        ("matched_terms.csv", matches),
        ("news_coverage.csv", summaries),
        ("model_deltas.csv", comparisons),
        ("control_deltas.csv", controls),
        ("translation_deltas.csv", translation_deltas),
        ("control_summary.csv", control_summaries),
    ):
        write_csv(name, values)
    manifest = dict(
        created_at_utc=datetime.now(UTC).isoformat(),
        samples=samples,
        resources=resources,
        downloads=downloads,
        nrc_v2=dict(
            path=NRC_V2_PATH.relative_to(PROJECT_ROOT).as_posix(),
            sha256=sha256(NRC_V2_PATH),
            member=NRC_V2_MEMBER,
        ),
        model_labels=MODEL_LABELS,
        input_path=INPUT_PATH.relative_to(PROJECT_ROOT).as_posix(),
        input_sha256=sha256(INPUT_PATH),
        script_sha256=sha256(Path(__file__)),
        matching=(
            "NFKC, casefold, Unicode word tokens, longest non-overlapping "
            "whitespace-separated match; no stemming, stopword removal, or "
            "negation handling."
        ),
        duplicate_policy=(
            "Average VAD values of rows mapping to the same normalized term before "
            "scoring; each text occurrence has equal weight."
        ),
        scaling=(
            "NRC v1 [0,1] -> 1+4*x; NRC v2.1 [-1,1] -> 3+2*x; "
            "MEmoLon [1,9] -> (x+1)/2; current model raw project logits. "
            "No clipping or sample-wise normalization."
        ),
        missing_policy="No lexical matches produce missing VAD, never neutral imputation.",
        limitation=(
            "Purposive diagnostic; no human document VAD ground truth or "
            "statistical accuracy ranking. Translation controls are manual, not a "
            "validated pipeline."
        ),
    )
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    control_lookup = {
        (row["model"], row["language"], row["kind"]): row for row in control_summaries
    }
    scope_lookup = {
        (row["model"], row["language"]): row
        for row in controls
        if row["kind"] == "negation_scope_control"
    }
    bert_word_pt = control_lookup[("current_model", "pt", "affect_word_control")]
    bert_word_en = control_lookup[("current_model", "en", "affect_word_control")]
    bert_neg_pt = control_lookup[("current_model", "pt", "negation_control")]
    bert_neg_en = control_lookup[("current_model", "en", "negation_control")]
    bert_scope_pt = scope_lookup[("current_model", "pt")]["expected_direction_delta"]
    bert_scope_en = scope_lookup[("current_model", "en")]["expected_direction_delta"]
    lines = [
        "# Portuguese and English comparison of four VAD scorers",
        "",
        (
            "This diagnostic compares NRC v1, MEmoLon MTL_grouped, NRC v2.1, and "
            "the current project model RobroKools/vad-bert on the same texts. "
            "It is not an accuracy benchmark."
        ),
        "",
        (
            "NRC v1 uses its official Portuguese translation and original English lexicon; "
            "MEmoLon uses language-specific Portuguese/English files. NRC v2.1 uses its "
            "English lexicon in both languages; Portuguese matches are lexical overlap, "
            "not validated Portuguese recognition. RobroKools/vad-bert is documented "
            "for English and Portuguese inference is an exploratory probe. English news "
            "controls use the same manual translations recorded in the previous comparison."
        ),
        "",
        (
            f"The current BERT model gives the expected direction in "
            f"{bert_word_pt['expected_direction_observed']}/3 Portuguese and "
            f"{bert_word_en['expected_direction_observed']}/3 English affect-word contrasts. "
            f"For simple negation contrasts the counts are "
            f"{bert_neg_pt['expected_direction_observed']}/3 in Portuguese and "
            f"{bert_neg_en['expected_direction_observed']}/3 in English. Its valence gap "
            f"on the same-word scope contrast is {bert_scope_pt:.4f} in Portuguese and "
            f"{bert_scope_en:.4f} in English. These chosen controls measure qualitative "
            "sensitivity only, not accuracy or general language competence."
        ),
        "",
        "## News coverage",
        "",
        "| Language | Model | Matched / total tokens | Coverage |",
        "| --- | --- | ---: | ---: |",
    ]
    for summary in summaries:
        coverage = summary["weighted_coverage"]
        matched = (
            f"{summary['matched_tokens']}/{summary['token_count']}"
            if coverage is not None
            else "not applicable"
        )
        coverage_label = f"{coverage:.2%}" if coverage is not None else "not applicable"
        lines.append(
            f"| {summary['language']} | {summary['model']} | {matched} | {coverage_label} |"
        )
    lines += [
        "",
        (
            "Coverage is pooled by token count, not averaged across documents. "
            "Portuguese input consists of the same three rescue, stadium accident, "
            "and hailstorm captions used in the earlier smoke test. English input "
            "consists of manual translations. These collected captions are not "
            "independently verified full articles. Hashtags, names, numbers, URLs, "
            "and function words remain in the token denominator. Lexical coverage is "
            "not applicable to the contextual neural model; it is never reported as 100%. "
            "Wordpiece lengths, unknown pieces, and truncation flags are in scores.csv."
        ),
        "",
        "## Per-sample news scores",
        "",
        "| Sample | Language | Model | Coverage | V | A | D |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row["kind"] in ("local_news_post", "manual_translation_control", "coverage_only_title"):
            scores = " | ".join(
                "missing" if row[dim] is None else f"{row[dim]:.3f}" for dim in DIMENSIONS
            )
            coverage_label = (
                f"{row['token_coverage']:.2%}"
                if row["token_coverage"] is not None
                else "not applicable"
            )
            lines.append(
                f"| {row['label']} | {row['language']} | {row['model']} | "
                f"{coverage_label} | {scores} |"
            )
    lines += [
        "",
        (
            "The TSE training headline is assessed for lexical coverage only, and "
            "has deliberately suppressed affect scores. No political actors or "
            "positions are compared."
        ),
        "",
        "## Control summary",
        "",
        "| Model | Language | Control | Expected direction / available | Missing |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for summary in control_summaries:
        lines.append(
            f"| {summary['model']} | {summary['language']} | {summary['kind']} | "
            f"{summary['expected_direction_observed']}/{summary['available']} | "
            f"{summary['missing']} |"
        )
    lines += [
        "",
        "Available excludes missing scores. A result of 0/0 means no evaluable contrast.",
        "",
        "## Diagnostic controls",
        "",
        "| Control | Language | Dimension | Model | Expected-direction delta | Observed? |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for control in controls:
        delta = control["expected_direction_delta"]
        lines.append(
            f"| {control['kind']} | {control['language']} | {control['dimension']} | "
            f"{control['model']} | {'missing' if delta is None else f'{delta:.3f}'} | "
            f"{control['expected_direction_observed']} |"
        )
    lines += [
        "",
        (
            "Scope controls compare `Estou feliz, mas não estou triste.` with "
            "`Estou triste, mas não estou feliz.`, and their English counterparts. "
            "These have the same word multiset but different negation scope. Their "
            "The table gives their valence differences for each scorer and language. "
            "Word-control deltas are high minus low for the target dimension. "
            "Negation deltas are affirmative minus negated. A positive delta is a "
            "qualitative diagnostic expectation, not a human numeric label. "
            "Passing a negation direction check does not establish scope "
            "understanding: adding a lexically scored negator can shift an average "
            "without changing the value of the negated adjective."
        ),
        "",
        "## Method and limits",
        "",
        (
            "All three lexicons use identical tokenization, longest-match rules, "
            "duplicate aggregation, occurrence weighting, and missing-value "
            "handling. NRC Portuguese can have several English source words "
            "translated to the same term; their ratings are averaged first to "
            "avoid arbitrary row-order choices. Multiword translations are "
            "supported. Unsupported lexical forms are counted in the manifest. "
            "Only input-relevant entries are retained in memory after streaming "
            "each complete lexicon. Case-insensitive normalization also merges "
            "repeated and differently cased MEmoLon rows. Their values are averaged "
            "by the same rule as NRC translations. The manifest counts input-relevant "
            "duplicate rows for both resources. A different case or duplicate policy "
            "could produce different scores."
        ),
        "",
        (
            "All reported VAD scores use the project's nominal 1-5 scale. NRC v1 "
            "uses 1+4*x from [0,1]; NRC v2.1 uses 3+2*x from [-1,1]; MEmoLon "
            "uses (x+1)/2 from [1,9]. BERT uses raw regression logits, exactly as "
            "the project does. No source or model outputs are clipped. Resource estimates "
            "outside their nominal native range are counted rather than silently "
            "clipped. Equal numeric scales do not imply equal calibration: the "
            "resources use different source norms and rating procedures."
        ),
        "",
        (
            "Model deltas and matched-term traces show what changes with the "
            "resource. Translation deltas measure sensitivity to this manual "
            "wording choice, not cross-language accuracy. News coverage, score "
            "agreement, and affect-control separation cannot identify the correct "
            "human document score. A held-out, human-rated Portuguese Brazilian "
            "and English sample is needed before adopting any scorer as more "
            "accurate."
        ),
        (
            "The current model is loaded from a cached, revision-identified Hugging Face "
            "snapshot and run with the existing project inference functions, on CPU in "
            "evaluation mode. The manifest records model hashes, package versions, batch "
            "size, and the 512-token truncation limit. No new translations or model tuning "
            "were performed for this extension."
        ),
        "",
        (
            "The production scorer, model default, simulation, STDI, and TCC files "
            "are unchanged. This audit does not integrate MEmoLon into the "
            "application. The model_deltas.csv schema now represents all six pairs; "
            "its columns are model_a, model_b, and signed delta_b_minus_a values, "
            "replacing the previous MEmoLon-minus-NRC-only columns."
        ),
        "",
        "## Sources and reproduction",
        "",
        (
            "- [NRC VAD v1 and automatic Portuguese "
            "translation](https://saifmohammad.com/WebPages/nrc-vad.html)."
        ),
        (
            "- [MEmoLon code and recommended MTL_grouped "
            "release](https://github.com/JULIELab/MEmoLon)."
        ),
        "- [MEmoLon data record](https://zenodo.org/records/3756607).",
        "- [MEmoLon paper](https://aclanthology.org/2020.acl-main.112/).",
        "- [Current model card](https://huggingface.co/RobroKools/vad-bert).",
        f"- [TSE training headline]({TSE_URL}).",
        "",
        (
            "Public resources are cached locally, not committed or redistributed "
            "as project data. Downloads use the official NRC ZIP and selected "
            "Portuguese/English members of the official MEmoLon ZIP via HTTP "
            "ranges. ZIP member CRCs and SHA256 are recorded; the complete 2.36 GB "
            "archive MD5 was not checked."
        ),
        "",
        "```powershell",
        ("uv run python scripts/download_vad_comparison_resources.py --include-portuguese-pilot"),
        (
            "uv --cache-dir .cache/uv run --offline --no-sync python "
            "scripts/compare_portuguese_vad_lexicons.py"
        ),
        "```",
        "",
        (
            "The download step needs network access; the comparison runs offline "
            "once the cache exists. NRC v2.1 also requires the existing archive in "
            "`.cache/nrc_vad/`, and BERT requires its existing Hugging Face snapshot. "
            "`manifest.json` records full texts, source "
            "IDs, resource statistics, and hashes. `scores.csv` contains all "
            "sample scores; `matched_terms.csv` contains occurrence-level lexical "
            "evidence for lexicons only; the remaining CSVs contain coverage and paired "
            "deltas. model_deltas.csv contains all six unordered model pairs and signed "
            "model_b minus model_a differences. control_summary.csv counts available "
            "and expected-direction contrasts separately, so missing is never a pass."
        ),
    ]
    (OUTPUT_DIR / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(dict(news_coverage=summaries, resources=resources, controls=controls), indent=2)
    )


if __name__ == "__main__":
    main()
