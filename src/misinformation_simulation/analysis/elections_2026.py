"""Local data preparation for the Brazilian elections exploratory notebook."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

POST_COLUMNS = [
    "id",
    "creation_time",
    "modified_time",
    "baixado_em_utc",
    "text",
    "hashtags",
    "content_type",
    "lang",
    "conta_id",
    "conta_usuario",
    "post_owner.name",
    "grupo",
    "cargo",
    "uf",
    "nome_urna",
    "sq_candidato",
    "partido",
    "campo_nacional",
    "campo_disputa",
    "lado7",
    "genero",
    "tipo_coleta",
    "arquivo",
    "mcl_url",
    "statistics.like_count",
    "statistics.comment_count",
    "statistics.views",
    "statistics.views_date_last_refreshed",
    "idade_horas",
    "n_downloads",
]
POST_METRICS = {
    "statistics.like_count": "likes",
    "statistics.comment_count": "comments",
    "statistics.views": "views",
}


def parquet_inventory(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    files, fields = [], []
    for path in sorted(data_dir.glob("posts_*.parquet")):
        parquet = pq.ParquetFile(path)
        files.append(
            {
                "source_file": path.name,
                "rows": parquet.metadata.num_rows,
                "size_mb": path.stat().st_size / 1_000_000,
            }
        )
        fields.extend(
            {"source_file": path.name, "column": field.name, "dtype": str(field.type)}
            for field in parquet.schema_arrow
        )
    if not files:
        raise FileNotFoundError(f"No posts_*.parquet files found in {data_dir}")
    return pd.DataFrame(files), pd.DataFrame(fields)


def load_posts(data_dir: Path) -> pd.DataFrame:
    """Read analytical columns only, preserving provenance and missing metrics."""
    frames = []
    for path in sorted(data_dir.glob("posts_*.parquet")):
        available = pq.ParquetFile(path).schema_arrow.names
        required = {"id", "text", "creation_time", "conta_usuario", "grupo"}
        if missing := required.difference(available):
            raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")
        frame = pd.read_parquet(path, columns=[c for c in POST_COLUMNS if c in available])
        frame["source_file"] = path.name
        frame["collection_period"] = path.stem.removeprefix("posts_")
        for column in POST_METRICS:
            if column in frame:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No posts_*.parquet files found in {data_dir}")
    posts = pd.concat(frames, ignore_index=True).rename(columns=POST_METRICS)
    for column in posts.select_dtypes(include=["object", "string"]):
        posts[column] = posts[column].astype("string").replace(r"^\s*$", pd.NA, regex=True)
    for column in ["creation_time", "modified_time", "baixado_em_utc"]:
        posts[f"{column}_utc"] = pd.to_datetime(posts[column], errors="coerce", utc=True)
    posts["published_at_local"] = posts["creation_time_utc"].dt.tz_convert("America/Sao_Paulo")
    posts["day_utc"] = posts["creation_time_utc"].dt.strftime("%Y-%m-%d")
    posts["day_local"] = posts["published_at_local"].dt.strftime("%Y-%m-%d")
    posts["word_count"] = posts["text"].fillna("").str.count(r"\b\w+\b")
    posts["char_count"] = posts["text"].fillna("").str.len()
    posts["text_hash"] = posts["text"].map(text_fingerprint).astype("string")
    posts["download_lag_hours"] = (
        posts["baixado_em_utc_utc"] - posts["creation_time_utc"]
    ).dt.total_seconds() / 3600
    for column in [
        "grupo",
        "cargo",
        "uf",
        "partido",
        "campo_nacional",
        "campo_disputa",
        "lado7",
        "genero",
        "content_type",
        "lang",
        "tipo_coleta",
        "source_file",
        "collection_period",
    ]:
        if column in posts:
            posts[column] = posts[column].astype("category")
    return posts


def text_fingerprint(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    normalized = " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest() if normalized else None


def quality_profile(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in frame:
        values = frame[column]
        missing = values.isna()
        if not pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_datetime64_any_dtype(
            values
        ):
            missing |= values.astype("string").str.strip().eq("").fillna(False)
        rows.append(
            {
                "column": column,
                "dtype": str(values.dtype),
                "missing_rows": int(missing.sum()),
                "missing_pct": 100 * missing.mean(),
                "unique_nonmissing": values[~missing].nunique(),
            }
        )
    return pd.DataFrame(rows).sort_values("missing_pct", ascending=False)


def distribution(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    labels = frame[column].astype("string").fillna("Unknown")
    result = labels.value_counts().rename_axis(column).reset_index(name="posts")
    result["share_pct"] = 100 * result["posts"] / len(frame) if len(frame) else 0
    if "conta_usuario" in frame:
        accounts = frame.assign(_label=labels).groupby("_label")["conta_usuario"].nunique()
        result["active_accounts"] = result[column].map(accounts)
    return result


def parse_json_records(value: object) -> list[dict]:
    """Accept JSON arrays, objects, and the ad CSV's comma-separated objects."""
    if not isinstance(value, str) or not value.strip():
        return []
    for candidate in [value, f"[{value}]"]:
        try:
            parsed = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    return []


def parse_hashtags(value: object) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except ValueError:
        return []
    if not isinstance(parsed, list):
        return []
    return sorted(
        {str(item).strip().lstrip("#").casefold() for item in parsed if str(item).strip()}
    )


def parse_bounds(value: object) -> tuple[float, float]:
    """Keep unknown or open interval endpoints missing instead of inventing midpoints."""
    if not isinstance(value, str):
        return float("nan"), float("nan")
    endpoints = []
    for key in ["lower_bound", "upper_bound"]:
        match = re.search(rf'["\x27]?{key}["\x27]?\s*:\s*["\x27]?(\d+(?:\.\d+)?)', value)
        endpoints.append(float(match.group(1)) if match else float("nan"))
    return tuple(endpoints)


def load_ads(data_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(data_dir.rglob("*.csv")):
        if "meta-ad-library" not in str(path.relative_to(data_dir)):
            continue
        frame = pd.read_csv(path, dtype="string")
        frame["source_file"] = path.relative_to(data_dir).as_posix()
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No Meta Ad Library CSV found under {data_dir}")
    ads = pd.concat(frames, ignore_index=True).replace(r"^\s*$", pd.NA, regex=True)
    for column in ["ad_creation_time", "ad_delivery_start_time", "ad_delivery_stop_time"]:
        ads[f"{column}_utc"] = pd.to_datetime(ads[column], errors="coerce", utc=True)
    for column in ["impressions", "spend", "estimated_audience_size"]:
        ads[[f"{column}_lower", f"{column}_upper"]] = pd.DataFrame(
            ads[column].map(parse_bounds).tolist(), index=ads.index
        )
    ads["text"] = ads["ad_creative_bodies"]
    ads["word_count"] = ads["text"].fillna("").str.count(r"\b\w+\b")
    return ads


def load_aggregate_package(data_dir: Path) -> tuple[dict[str, pd.DataFrame], str]:
    paths = sorted(data_dir.glob("brpol_agregados_*.zip"))
    if len(paths) != 1:
        raise ValueError(f"Expected one aggregate ZIP, found {len(paths)}")
    tables, readme = {}, ""
    with zipfile.ZipFile(paths[0]) as archive:
        for name in archive.namelist():
            if name.endswith(".csv"):
                with archive.open(name) as stream:
                    tables[Path(name).stem] = pd.read_csv(stream, dtype="string")
            elif name.endswith("README.md"):
                readme = archive.read(name).decode("utf-8")
    return tables, readme


def build_seed_pool(
    posts: pd.DataFrame,
    *,
    groups: list[str],
    min_words: int = 80,
    max_words: int = 1200,
    languages: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build review candidates, without assuming captions are complete news articles."""
    languages = ["pt"] if languages is None else languages
    pool = posts.copy()
    audit = [{"step": "All loaded posts", "remaining": len(pool)}]
    conditions = [
        ("Requested source groups", pool["grupo"].isin(groups)),
        ("Requested language labels", pool["lang"].isin(languages)),
        (
            "Valid post ID, account and publication date",
            pool["id"].notna() & pool["conta_usuario"].notna() & pool["creation_time_utc"].notna(),
        ),
        ("Exclude stories", pool["content_type"].ne("stories")),
        ("Configured text length", pool["word_count"].between(min_words, max_words)),
    ]
    for label, mask in conditions:
        pool = pool.loc[mask.reindex(pool.index).fillna(False)]
        audit.append({"step": label, "remaining": len(pool)})
    pool = pool.sort_values(["creation_time_utc", "source_file", "id"])
    for column, label in [("id", "Unique post IDs"), ("text_hash", "Unique normalized text")]:
        pool = pool.drop_duplicates(column, keep="first")
        audit.append({"step": label, "remaining": len(pool)})
    return pool.reset_index(drop=True), pd.DataFrame(audit)


def sample_seed_pool(
    pool: pd.DataFrame,
    *,
    per_stratum: int = 10,
    max_per_account: int = 2,
    seed: int = 2026,
    strata: tuple[str, ...] = ("grupo", "collection_period"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Draw up to the requested quota per stratum with a global account cap."""
    if per_stratum < 1 or max_per_account < 1 or not strata:
        raise ValueError("Quotas, account cap, and strata must be positive/nonempty")
    shuffled = pool.sort_values("id").sample(frac=1, random_state=seed)
    account_counts, stratum_counts, selected = {}, {}, []
    for index, row in shuffled.iterrows():
        key = tuple("Unknown" if pd.isna(row[c]) else str(row[c]) for c in strata)
        account = str(row["conta_usuario"])
        if stratum_counts.get(key, 0) >= per_stratum:
            continue
        if account_counts.get(account, 0) >= max_per_account:
            continue
        selected.append(index)
        stratum_counts[key] = stratum_counts.get(key, 0) + 1
        account_counts[account] = account_counts.get(account, 0) + 1
    sample = pool.loc[selected].copy().reset_index(drop=True)
    availability = (
        pool.groupby(list(strata), observed=True, dropna=False).size().rename("available")
    )
    chosen = sample.groupby(list(strata), observed=True, dropna=False).size().rename("selected")
    audit = pd.concat([availability, chosen], axis=1).fillna(0).reset_index()
    audit["selected"] = audit["selected"].astype(int)
    audit["requested"] = per_stratum
    audit["shortfall"] = (per_stratum - audit["selected"]).clip(lower=0)
    sample["description"] = sample["text"]
    sample["category"] = sample["grupo"].astype("string")
    sample["source_post_id"] = sample["id"]
    sample["article_id"] = "brpol2026_" + sample["id"].astype("string")
    sample["manual_review_status"] = "pending"
    return sample, audit


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
