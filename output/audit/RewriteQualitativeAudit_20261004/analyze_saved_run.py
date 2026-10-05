"""Summarize saved rewrites without generating or rescoring any text."""

import csv
import hashlib
import itertools
import json
import random
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "output/interaction_graph/app_runs/simulation_ui_20261004_183721"
OUT = Path(__file__).resolve().parent
COMPONENTS = ("theme", "subtopic", "entity", "relation", "contradiction", "vad")


def write_csv(name, records):
    with (OUT / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


chains = {}
inputs = []
for directory in sorted(RUN.iterdir()):
    path = next(directory.glob("*_steps.jsonl"))
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    chains[directory.name] = rows
    for source in (path, next(directory.glob("*_summary.json"))):
        inputs.append({"path": source.relative_to(ROOT).as_posix(),
                       "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})

news_ids = [r["news_id"] for r in next(iter(chains.values())) if r["step_index"] == 1]
assert len(news_ids) == len(set(news_ids)) == 20
summary = []
per_news = []
review = ["# Saved messages for qualitative review", "",
          "All 20 source descriptions and 160 saved outputs. No new generation or scoring.", ""]
finals = {}
for name, rows in chains.items():
    assert len(rows) == 40
    assert all(r["rewrite_status"] == "success" for r in rows)
    finals[name] = {r["news_id"]: r for r in rows if r["step_index"] == 2}
    for step in (1, 2):
        subset = [r for r in rows if r["step_index"] == step]
        record = {"chain": name, "step": step, "n": len(subset),
                  "persona": subset[0]["node_label"],
                  "mean_stdi_vs_original": statistics.mean(r["stdi_vs_original"] for r in subset),
                  "sd_stdi_vs_original": statistics.stdev(r["stdi_vs_original"] for r in subset),
                  "mean_stdi_incremental": statistics.mean(r["stdi_incremental"] for r in subset),
                  "mean_output_words": statistics.mean(len(r["rewritten_text"].split()) for r in subset)}
        for component in COMPONENTS:
            key = component + "_drift_vs_original"
            record["mean_" + key] = statistics.mean(r[key] for r in subset)
        record["mean_vad_stdi_contribution"] = statistics.mean(
            r["stdi_vs_original"] - (r["content_drift_vs_original"] +
                (1 - r["content_drift_vs_original"]) * .2 * r["contradiction_drift_vs_original"])
            for r in subset)
        summary.append(record)

structure_variants = {}
for number, news_id in enumerate(news_ids, 1):
    matching = {name: [(line, r) for line, r in enumerate(rows, 1)
                       if r["news_id"] == news_id] for name, rows in chains.items()}
    originals = {values[0][1]["source_text"] for values in matching.values()}
    assert len(originals) == 1
    structure_variants[news_id] = len({values[0][1]["metadata_original_json"]
                                      for values in matching.values()})
    first = next(iter(matching.values()))[0][1]
    review += [f"## {number}. {first['metadata_title']}", "",
               f"News ID: `{news_id}`", "", "### Original description", "",
               first["source_text"], ""]
    for name, values in matching.items():
        assert values[1][1]["source_text"] == values[0][1]["rewritten_text"]
        for line, row in values:
            review += [f"### {name}: step {row['step_index']} ({row['node_label']})", "",
                       f"Source JSONL line: {line}. STDI vs original: {row['stdi_vs_original']:.6f}; "
                       f"incremental: {row['stdi_incremental']:.6f}.", "",
                       row["rewritten_text"], ""]
        row = values[1][1]
        per_news.append({"news_number": number, "news_id": news_id,
                         "title": first["metadata_title"], "chain": name,
                         "source_jsonl_line": values[1][0],
                         "first_stdi_vs_original": values[0][1]["stdi_vs_original"],
                         "final_stdi_vs_original": row["stdi_vs_original"],
                         "second_stdi_incremental": row["stdi_incremental"],
                         "original_structure_variants": structure_variants[news_id]})

paired = []
generator = random.Random(42)
for left, right in itertools.combinations(chains, 2):
    differences = [finals[left][nid]["stdi_vs_original"] -
                   finals[right][nid]["stdi_vs_original"] for nid in news_ids]
    samples = sorted(statistics.mean(generator.choices(differences, k=20))
                     for _ in range(10000))
    paired.append({"left_chain": left, "right_chain": right, "n_news": 20,
                   "mean_paired_final_difference": statistics.mean(differences),
                   "bootstrap_95_lower": samples[249], "bootstrap_95_upper": samples[9749],
                   "left_greater_count": sum(d > 0 for d in differences)})

write_csv("chain_summary.csv", summary)
write_csv("per_news_final_scores.csv", per_news)
write_csv("paired_final_comparisons.csv", paired)
(OUT / "review_texts.md").write_text("\n".join(review), encoding="utf-8")
manifest = {"run": RUN.relative_to(ROOT).as_posix(), "n_news": 20, "n_outputs": 160,
            "all_original_descriptions_match": True, "all_relays_match": True,
            "original_structure_variants_by_news": structure_variants,
            "bootstrap": {"replicates": 10000, "seed": 42, "unit": "paired news",
                          "method": "percentile", "multiple_comparison_adjustment": False,
                          "limitation": "Does not capture generation or extraction rerun uncertainty."},
            "world_fact_check_performed": False, "new_llm_calls": 0,
            "source_files": inputs}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps({"summary": summary, "paired_comparisons": paired}, indent=2))
