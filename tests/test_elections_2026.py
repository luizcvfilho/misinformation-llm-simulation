import json
import math
import zipfile

import pandas as pd
import pytest

from misinformation_simulation.analysis.elections_2026 import (
    build_seed_pool,
    load_ads,
    load_aggregate_package,
    load_posts,
    parse_bounds,
    parse_hashtags,
    parse_json_records,
    sample_seed_pool,
    text_fingerprint,
)


def test_ad_nested_records_accept_export_format_and_reject_malformed_input():
    records = [{"region": "São Paulo", "percentage": 0.75}, {"region": "RJ", "percentage": 0.25}]
    export = ",".join(json.dumps(record) for record in records)
    assert parse_json_records(export) == records
    assert parse_json_records(json.dumps(records)) == records
    assert parse_json_records(json.dumps(records[0])) == records[:1]
    assert parse_json_records("malformed") == []
    assert parse_json_records(pd.NA) == []


def test_open_ad_bounds_do_not_become_zero_or_point_estimates():
    assert parse_bounds("lower_bound: 0, upper_bound: 999") == (0, 999)
    lower, upper = parse_bounds("lower_bound: 1000001")
    assert lower == 1000001
    assert math.isnan(upper)
    assert all(math.isnan(endpoint) for endpoint in parse_bounds(pd.NA))
    assert parse_bounds('{"lower_bound": "100", "upper_bound": "199"}') == (100, 199)


def test_caption_fingerprints_and_hashtags_do_not_inflate_duplicate_counts():
    assert text_fingerprint("  ELEIÇÕES\n2026  ") == text_fingerprint("eleições 2026")
    assert text_fingerprint("ação") != text_fingerprint("acao")
    assert text_fingerprint(" \n ") is None
    assert parse_hashtags('["#Vote", "vote", "Brasil"]') == ["brasil", "vote"]
    assert parse_hashtags("malformed") == []


def test_post_loading_handles_mixed_schemas_missing_counts_and_local_boundary(tmp_path):
    base = {
        "id": ["9007199254740993"],
        "text": ["A standalone caption"],
        "creation_time": ["2026-08-15T01:00:00Z"],
        "modified_time": ["2026-08-15T01:01:00Z"],
        "baixado_em_utc": ["2026-08-16T01:00:00Z"],
        "conta_usuario": ["account"],
        "grupo": ["veiculo_nacional"],
        "content_type": ["photos"],
        "lang": ["pt"],
        "statistics.like_count": [""],
        "statistics.comment_count": ["3"],
        "statistics.views": ["invalid"],
    }
    pd.DataFrame(base).to_parquet(tmp_path / "posts_first.parquet", index=False)
    second = {
        **base,
        "id": ["9007199254740994"],
        "statistics.like_count": [0],
        "statistics.comment_count": [4],
        "statistics.views": [10],
    }
    pd.DataFrame(second).to_parquet(tmp_path / "posts_second.parquet", index=False)
    posts = load_posts(tmp_path)
    assert posts["id"].tolist() == ["9007199254740993", "9007199254740994"]
    assert pd.isna(posts.loc[0, "likes"])
    assert posts.loc[1, "likes"] == 0
    assert pd.isna(posts.loc[0, "views"])
    assert posts["day_utc"].tolist() == ["2026-08-15"] * 2
    assert posts["day_local"].tolist() == ["2026-08-14"] * 2
    assert posts["download_lag_hours"].tolist() == [24] * 2


def test_ad_loading_finds_nested_csv_and_preserves_identifiers(tmp_path):
    nested = tmp_path / "meta-ad-library-23" / "09"
    nested.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "ad_archive_id": "9007199254740993",
                "page_id": "00012",
                "ad_creation_time": "2026-09-20",
                "ad_delivery_start_time": "2026-09-21",
                "ad_delivery_stop_time": "",
                "ad_creative_bodies": "Original text",
                "impressions": "lower_bound: 0, upper_bound: 999",
                "spend": "lower_bound: 100, upper_bound: 199",
                "estimated_audience_size": "lower_bound: 1000001",
            }
        ]
    ).to_csv(nested / "2026.csv", index=False)
    ads = load_ads(tmp_path)
    assert ads.loc[0, "page_id"] == "00012"
    assert ads.loc[0, "ad_archive_id"] == "9007199254740993"
    assert pd.isna(ads.loc[0, "ad_delivery_stop_time_utc"])
    assert pd.isna(ads.loc[0, "estimated_audience_size_upper"])
    assert ads.loc[0, "impressions_lower"] == 0


def test_simulation_sampling_filters_duplicates_and_reports_account_cap_shortfall():
    rows = []
    for index in range(12):
        text = f"Original caption number {index} with useful context"
        rows.append(
            {
                "id": str(index),
                "text": text,
                "text_hash": text_fingerprint(text),
                "grupo": "veiculo_nacional",
                "lang": "pt",
                "content_type": "photos",
                "creation_time_utc": pd.Timestamp("2026-08-16", tz="UTC"),
                "source_file": "posts.parquet",
                "word_count": 8,
                "collection_period": "first" if index < 6 else "second",
                "conta_usuario": "same_account",
            }
        )
    rows[1]["text_hash"] = rows[0]["text_hash"]
    rows[2]["content_type"] = "stories"
    rows[3]["lang"] = "en"
    posts = pd.DataFrame(rows)
    pool, audit = build_seed_pool(posts, groups=["veiculo_nacional"], min_words=5)
    assert len(pool) == 9
    assert audit["remaining"].is_monotonic_decreasing
    sample, quotas = sample_seed_pool(pool, per_stratum=3, max_per_account=2, seed=2026)
    repeated, _ = sample_seed_pool(
        pool.sample(frac=1, random_state=1), per_stratum=3, max_per_account=2, seed=2026
    )
    assert sample["id"].tolist() == repeated["id"].tolist()
    assert len(sample) == 2
    assert quotas["shortfall"].sum() == 4
    assert sample["description"].tolist() == sample["text"].tolist()
    assert sample["manual_review_status"].eq("pending").all()
    with pytest.raises(ValueError):
        sample_seed_pool(pool, per_stratum=0)


def test_aggregate_zip_loads_csv_tables_and_readme_without_extracting(tmp_path):
    archive_path = tmp_path / "brpol_agregados_test.zip"
    names = ["contas", "conta_dia", "agregado_dia", "hashtags", "candidatos_tse"]
    with zipfile.ZipFile(archive_path, "w") as archive:
        for name in names:
            archive.writestr(f"package/{name}.csv", "identifier,label\n00012,Eleições\n")
        archive.writestr("package/README.md", "# Source notes\nCollection documentation.")
    tables, readme = load_aggregate_package(tmp_path)
    assert set(tables) == set(names)
    assert all(frame.loc[0, "identifier"] == "00012" for frame in tables.values())
    assert all(frame.loc[0, "label"] == "Eleições" for frame in tables.values())
    assert "Collection documentation." in readme
    assert list(tmp_path.iterdir()) == [archive_path]
