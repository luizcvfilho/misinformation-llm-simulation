"""Reproduce Chapter 5 tables and figures from the six frozen final runs.

Run with ``uv run --with matplotlib python scripts/analyze_tcc_chapter_5.py``.
No model is loaded and no API is called. All analyses are descriptive.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/analysis/tcc_chapter_5"
FIGURES = ROOT / "tcc/figuras/chapter5"
RUNS = [
    ("English", "GPT", "simulation_ui_20261016_213859-GPT"),
    ("English", "Grok", "simulation_ui_20261017_160546-GROK"),
    ("English", "Gemini", "simulation_ui_20261017_221432-Gemini"),
    ("Portuguese", "GPT", "simulation_ui_20261018_172257-GPT-BR"),
    ("Portuguese", "Grok", "simulation_ui_20261018_172257-GROK-BR"),
    ("Portuguese", "Gemini", "simulation_ui_20261019_124942-GEMINIBR"),
]
CHAINS = ["NNNN", "CCCC", "PPPP", "DDDD", "CCPP", "PPCC", "DDNN", "NNDD"]
CORPORA = ["English", "Portuguese"]
GENERATORS = ["GPT", "Grok", "Gemini"]
COMPONENTS = ["theme", "subtopic", "entity", "relation", "contradiction", "vad"]
COMPONENT_LABELS = ["Theme", "Subtopics", "Entities", "Relations", "Contradiction", "VAD"]
CONTRASTS = [
    ("NNDD", "NNNN"),
    ("CCPP", "CCCC"),
    ("PPCC", "PPPP"),
    ("DDNN", "DDDD"),
    ("CCPP", "PPCC"),
    ("NNDD", "DDNN"),
]
NUMBER = re.compile(r"(?<!\w)\d+(?:[.,]\d+)*(?:\s*%)?", re.UNICODE)
ATTRIBUTION = re.compile(
    r"\b(?:according to|said|says|say|stated|claims?|claimed|alleges?|alleged|"
    r"accuses?|accused|reports?|reported|segundo|afirma|afirmou|afirmaram|disse|"
    r"diz|dizem|declarou|alega|alegou|acusou|acusa|informou|relatou|aponta|apontou)\b",
    re.IGNORECASE,
)
UNCERTAINTY = re.compile(
    r"\b(?:may|might|could|possibly|possible|potential|reportedly|allegedly|"
    r"suspected|suspects?|uncertain|unconfirmed|pode|podem|poderia|poderiam|"
    r"possível|possíveis|possibilidade|suposto|suposta|supostos|supostas|"
    r"suspeita|suspeitas|suspeito|suspeitos|teria|teriam|talvez|hipótese)\b",
    re.IGNORECASE,
)
COLORS = dict(
    zip(
        CHAINS,
        ["#626f80", "#b73c3c", "#3168b6", "#9a47b0", "#e4883c", "#2d9a8c", "#ab7e32", "#df6e9b"],
        strict=True,
    )
)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def numeric_tokens(text: str) -> set[str]:
    return {re.sub(r"\s+", "", match) for match in NUMBER.findall(text)}


def write_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(OUT / f"{name}.csv", index=False, float_format="%.9f")


def description(values: pd.Series) -> dict:
    values = values.dropna().astype(float)
    return {
        "n": len(values),
        "mean": values.mean(),
        "median": values.median(),
        "q25": values.quantile(0.25),
        "q75": values.quantile(0.75),
        "min": values.min(),
        "max": values.max(),
    }


def rank_correlation(left: pd.Series, right: pd.Series) -> float:
    valid = pd.concat([left, right], axis=1).dropna()
    if len(valid) < 3 or valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(spearmanr(valid.iloc[:, 0], valid.iloc[:, 1]).statistic)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    rows, sources, originals = [], [], {}
    for corpus, generator, run in RUNS:
        run_dir = ROOT / "output/interaction_graph/app_runs" / run
        paths = sorted(run_dir.glob("*/*_steps.jsonl"))
        assert len(paths) == 4, (run, paths)
        for path in paths:
            sources.append({"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path)})
            with path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    raw = json.loads(line)
                    assert raw["rewrite_status"] == "success"
                    assert raw["stdi_status_vs_original"] == "valid"
                    assert raw["stdi_status_incremental"] == "valid"
                    news_id = str(raw["news_id"])
                    original = raw["metadata_original_text"]
                    rewritten = raw["rewritten_text"]
                    prefix = raw["node_id"].removeprefix("prefix_")
                    assert len(prefix) == int(raw["step_index"])
                    key = corpus, news_id
                    if key not in originals:
                        structure = json.loads(raw["metadata_original_json"])
                        originals[key] = {
                            "corpus": corpus,
                            "news_id": news_id,
                            "title": raw.get("metadata_title", ""),
                            "original_text": original,
                            "original_words": len(original.split()),
                            "numeric_tokens": len(numeric_tokens(original)),
                            "has_numbers": bool(NUMBER.search(original)),
                            "attribution_cues": len(ATTRIBUTION.findall(original)),
                            "uncertainty_cues": len(UNCERTAINTY.findall(original)),
                            "has_attribution": bool(ATTRIBUTION.search(original)),
                            "has_uncertainty": bool(UNCERTAINTY.search(original)),
                            "entities": len(structure["central_entities"]),
                            "relations": len(structure["central_relations"]),
                            "feature_reference_file": path.relative_to(ROOT).as_posix(),
                            "feature_reference_line": line_number,
                        }
                    assert originals[key]["original_text"] == original
                    tokens = numeric_tokens(original)
                    current_tokens = numeric_tokens(rewritten)
                    row = {
                        "corpus": corpus,
                        "generator": generator,
                        "run": run,
                        "news_id": news_id,
                        "tree": prefix[0],
                        "prefix": prefix,
                        "node_id": raw["node_id"],
                        "parent_node": raw["source_node_id"],
                        "iteration": len(prefix),
                        "original": raw["stdi_vs_original"],
                        "incremental": raw["stdi_incremental"],
                        "cumulative": raw["stdi_cumulative"],
                        "llm_original": raw["stdi_llm_judge_vs_original"],
                        "llm_incremental": raw["stdi_llm_judge_incremental"],
                        "word_ratio": len(rewritten.split()) / len(original.split()),
                        "rewritten_words": len(rewritten.split()),
                        "numeric_retention": len(tokens & current_tokens) / len(tokens)
                        if tokens
                        else np.nan,
                        "introduced_numeric_tokens": len(current_tokens - tokens),
                        "source_file": path.relative_to(ROOT).as_posix(),
                        "source_line": line_number,
                    }
                    for component in COMPONENTS:
                        for reference in ["vs_original", "incremental"]:
                            row[f"{component}_{reference}"] = raw[f"{component}_drift_{reference}"]
                        row[f"llm_{component}"] = raw[f"{component}_drift_llm_judge_vs_original"]
                    for dimension in ["valence", "arousal", "dominance"]:
                        row[f"llm_signed_{dimension}"] = (
                            raw[f"vad_llm_rewritten_{dimension}"]
                            - raw[f"vad_llm_original_{dimension}"]
                        )
                    if corpus == "English":
                        assert raw["stdi_vad_source_cluster_vs_original"] == "model"
                        assert raw["stdi_vad_source_llm_judge_vs_original"] == "llm"
                        for reference, short in [
                            ("vs_original", "original"),
                            ("incremental", "incremental"),
                        ]:
                            cluster = raw[f"cluster_evaluation_{reference}"]
                            judge = raw[f"llm_judge_evaluation_{reference}"]
                            assert cluster["status"] == judge["status"] == "valid"
                            common_cluster = cluster.get("dual_metrics")
                            common_judge = judge.get("dual_metrics")
                            if common_cluster is None or common_judge is None:
                                reference_text = (
                                    original if short == "original" else raw["source_text"]
                                )
                                assert reference_text == rewritten and judge["identity_shortcut"]
                                assert all(value == 0 for value in cluster["metrics"].values())
                                assert all(value == 0 for value in judge["metrics"].values())
                                common_cluster, common_judge = cluster["metrics"], judge["metrics"]
                            np.testing.assert_allclose(
                                common_cluster["vad_drift"],
                                common_judge["vad_drift"],
                                atol=1e-9,
                            )
                            row[f"cluster_{short}"] = cluster["metrics"]["stdi"]
                            row[f"cluster_common_{short}"] = common_cluster["stdi"]
                            row[f"llm_common_{short}"] = common_judge["stdi"]
                            np.testing.assert_allclose(
                                raw[f"stdi_{reference}"],
                                (row[f"cluster_common_{short}"] + row[f"llm_common_{short}"]) / 2,
                                atol=1e-10,
                            )
                        for component in COMPONENTS:
                            row[f"cluster_{component}"] = raw[
                                f"{component}_drift_cluster_vs_original"
                            ]
                    else:
                        np.testing.assert_allclose(row["original"], row["llm_original"])
                        np.testing.assert_allclose(row["incremental"], row["llm_incremental"])
                    rows.append(row)
        print(f"Loaded {corpus} / {generator}: 1,200 unique nodes", flush=True)
    nodes = pd.DataFrame(rows)
    features = pd.DataFrame(originals.values())
    assert len(nodes) == 7200 and len(features) == 100
    assert not nodes.duplicated(["run", "news_id", "node_id"]).any()
    for _, group in nodes.groupby("run"):
        assert len(group) == 1200 and group.news_id.nunique() == 50
    input_files = {
        "English": ROOT / "data/graphs/graph_politics_news.csv",
        "Portuguese": ROOT / "data/elections_2026/simulation_news_100_cleaned.csv",
    }
    for corpus, input_file in input_files.items():
        source = pd.read_csv(input_file, dtype=str, keep_default_na=False)
        source = source[source.article_id != "__query_metadata__"].head(50)
        observed = features[features.corpus == corpus].set_index("news_id")
        assert set(source.article_id) == set(observed.index)
        for _, entry in source.iterrows():
            assert entry.description.strip() == observed.loc[entry.article_id, "original_text"]
    return nodes, features, sources


def project_paths(nodes: pd.DataFrame) -> pd.DataFrame:
    paths = []
    for chain in CHAINS:
        subset = nodes[nodes.prefix.isin([chain[:i] for i in range(1, 5)])].copy()
        subset["chain"] = chain
        paths.append(subset)
    result = pd.concat(paths, ignore_index=True).sort_values(
        ["corpus", "generator", "news_id", "chain", "iteration"]
    )
    assert len(result) == 9600
    groups = ["corpus", "generator", "news_id", "chain"]
    assert result.groupby(groups).size().eq(4).all()
    for metric in ["original", "llm_original"]:
        result[f"delta_{metric}"] = result.groupby(groups)[metric].diff()
    for _, group in result.groupby(groups):
        np.testing.assert_allclose(group.cumulative, group.incremental.cumsum(), atol=1e-7)
    for left, right in CONTRASTS[:4]:
        a = result[(result.chain == left) & (result.iteration <= 2)]
        b = result[(result.chain == right) & (result.iteration <= 2)]
        keys = ["corpus", "generator", "news_id", "iteration"]
        paired = a.merge(b, on=keys, suffixes=("_a", "_b"), validate="one_to_one")
        assert paired.node_id_a.eq(paired.node_id_b).all()
        assert paired.original_a.eq(paired.original_b).all()
    return result


def summarize(paths: pd.DataFrame, features: pd.DataFrame) -> dict[str, pd.DataFrame]:
    final = paths[paths.iteration == 4].copy()
    grouping = ["corpus", "generator", "chain"]
    trajectory = []
    for key, group in paths.groupby(grouping + ["iteration"]):
        for metric in [
            "original",
            "incremental",
            "delta_original",
            "llm_original",
            "llm_incremental",
            "cumulative",
        ]:
            trajectory.append(
                dict(zip(grouping + ["iteration"], key, strict=True))
                | {"metric": metric}
                | description(group[metric])
            )
    contrast_rows, generator_rows = [], []
    for (corpus, generator), group in final.groupby(["corpus", "generator"]):
        for metric in ["original", "incremental", "llm_original", "llm_incremental"]:
            wide = group.pivot(index="news_id", columns="chain", values=metric)
            for left, right in CONTRASTS:
                difference = wide[left] - wide[right]
                contrast_rows.append(
                    {
                        "corpus": corpus,
                        "generator": generator,
                        "contrast": f"{left}-{right}",
                        "metric": metric,
                        "fraction_positive": (difference > 1e-9).mean(),
                        "fraction_negative": (difference < -1e-9).mean(),
                    }
                    | description(difference)
                )
    for (corpus, chain), group in final.groupby(["corpus", "chain"]):
        for metric in ["original", "llm_original"]:
            wide = group.pivot(index="news_id", columns="generator", values=metric)
            for left, right in [("Grok", "GPT"), ("Gemini", "GPT"), ("Grok", "Gemini")]:
                generator_rows.append(
                    {
                        "corpus": corpus,
                        "chain": chain,
                        "contrast": f"{left}-{right}",
                        "metric": metric,
                    }
                    | description(wide[left] - wide[right])
                )
    method_rows = []
    english = paths[paths.corpus == "English"]
    for (generator, chain, iteration), group in english.groupby(
        ["generator", "chain", "iteration"]
    ):
        for reference in ["original", "incremental"]:
            for method, left, right in [
                ("paired_estimators", f"cluster_{reference}", f"llm_{reference}"),
                ("common_dual_vad", f"cluster_common_{reference}", f"llm_common_{reference}"),
            ]:
                difference = group[left] - group[right]
                method_rows.append(
                    {
                        "generator": generator,
                        "chain": chain,
                        "iteration": iteration,
                        "reference": reference,
                        "comparison": method,
                        "spearman": rank_correlation(group[left], group[right]),
                        "mean_absolute_difference": difference.abs().mean(),
                        "fraction_cluster_higher": (difference > 1e-9).mean(),
                    }
                    | description(difference)
                )
    # One outcome per article and generator: each of the eight final paths has equal weight.
    article = final.groupby(["corpus", "generator", "news_id"], as_index=False).agg(
        outcome=("llm_original", "mean"),
        word_ratio=("word_ratio", "mean"),
        numeric_retention=("numeric_retention", "mean"),
    )
    article = article.merge(features, on=["corpus", "news_id"], validate="many_to_one")
    association_rows = []
    continuous = [
        "original_words",
        "numeric_tokens",
        "attribution_cues",
        "uncertainty_cues",
        "entities",
        "relations",
        "word_ratio",
    ]
    binary = ["has_numbers", "has_attribution", "has_uncertainty"]
    for (corpus, generator), group in article.groupby(["corpus", "generator"]):
        for feature in continuous:
            x = group[feature].rank().to_numpy()
            y = group.outcome.rank().to_numpy()
            length = group.original_words.rank().to_numpy()
            design = np.column_stack([np.ones(len(length)), length])
            rx = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
            ry = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
            adjusted = float(np.corrcoef(rx, ry)[0, 1]) if np.std(rx) > 1e-8 else np.nan
            association_rows.append(
                {
                    "corpus": corpus,
                    "generator": generator,
                    "feature": feature,
                    "n": len(group),
                    "spearman": rank_correlation(group[feature], group.outcome),
                    "length_adjusted_rank_correlation": adjusted,
                }
            )
        for feature in binary:
            yes, no = group[group[feature]], group[~group[feature]]
            association_rows.append(
                {
                    "corpus": corpus,
                    "generator": generator,
                    "feature": feature,
                    "n": len(group),
                    "n_present": len(yes),
                    "n_absent": len(no),
                    "mean_present": yes.outcome.mean(),
                    "mean_absent": no.outcome.mean(),
                    "mean_difference": yes.outcome.mean() - no.outcome.mean(),
                }
            )
    # Preserve per-path checks, rather than interpreting the pooled article result alone.
    path_associations = []
    enriched = final.merge(features, on=["corpus", "news_id"], validate="many_to_one")
    for key, group in enriched.groupby(grouping):
        for feature in continuous:
            path_associations.append(
                dict(zip(grouping, key, strict=True))
                | {
                    "feature": feature,
                    "spearman": rank_correlation(group[feature], group.llm_original),
                }
            )
    component_rows = []
    for key, group in final.groupby(grouping):
        for component in COMPONENTS:
            component_rows.append(
                dict(zip(grouping, key, strict=True))
                | {"component": component}
                | description(group[f"{component}_vs_original"])
            )
    transition_rows = []
    for key, group in paths[paths.iteration >= 2].groupby(grouping + ["iteration"]):
        transition_rows.append(
            dict(zip(grouping + ["iteration"], key, strict=True))
            | {
                "decreased": (group.delta_original < -1e-9).sum(),
                "increased": (group.delta_original > 1e-9).sum(),
                "unchanged": (group.delta_original.abs() <= 1e-9).sum(),
            }
        )
    return {
        "trajectory_summary": pd.DataFrame(trajectory),
        "paired_contrasts": pd.DataFrame(contrast_rows),
        "generator_contrasts": pd.DataFrame(generator_rows),
        "method_comparison": pd.DataFrame(method_rows),
        "article_outcomes": article,
        "feature_associations": pd.DataFrame(association_rows),
        "feature_associations_by_chain": pd.DataFrame(path_associations),
        "component_summary": pd.DataFrame(component_rows),
        "transition_counts": pd.DataFrame(transition_rows),
    }


def axes_grid() -> tuple:
    fig, axes = plt.subplots(2, 3, figsize=(9.4, 6.4), sharex=True, sharey=True)
    for i, corpus in enumerate(CORPORA):
        for j, generator in enumerate(GENERATORS):
            axes[i, j].set_title(f"{corpus} | {generator}")
            axes[i, j].grid(alpha=0.16, axis="y")
    return fig, axes


def save_figure(fig: plt.Figure, name: str, legend: bool = False) -> None:
    if legend:
        handles = [plt.Line2D([], [], color=COLORS[chain], label=chain) for chain in CHAINS]
        fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.05 if legend else 0, 1, 1))
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_trajectories(paths: pd.DataFrame, metric: str, name: str, ylabel: str) -> None:
    fig, axes = axes_grid()
    for i, corpus in enumerate(CORPORA):
        for j, generator in enumerate(GENERATORS):
            ax = axes[i, j]
            subset = paths[(paths.corpus == corpus) & (paths.generator == generator)]
            for chain in CHAINS:
                group = subset[subset.chain == chain].groupby("iteration")[metric]
                median = group.median()
                ax.plot(median.index, median, color=COLORS[chain], marker="o", ms=3, lw=1.5)
                ax.fill_between(
                    median.index,
                    group.quantile(0.25),
                    group.quantile(0.75),
                    color=COLORS[chain],
                    alpha=0.055,
                )
            ax.set_xticks([1, 2, 3, 4])
            if metric.startswith("delta"):
                ax.axhline(0, color="black", lw=0.7)
            else:
                ax.set_ylim(0, 0.85)
            if i == 1:
                ax.set_xlabel("Iteration")
            if j == 0:
                ax.set_ylabel(ylabel)
    save_figure(fig, name, legend=True)


def create_figures(paths: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    for metric, name, label in [
        ("original", "original_trajectories", "STDI vs original"),
        ("incremental", "incremental_trajectories", "Incremental STDI"),
        ("delta_original", "original_change", "Change in STDI vs original"),
        ("llm_original", "corpus_llm_trajectories", "LLM STDI vs original"),
    ]:
        plot_trajectories(paths, metric, name, label)
    final = paths[paths.iteration == 4]
    rng = np.random.default_rng(20261019)
    fig, axes = axes_grid()
    for i, corpus in enumerate(CORPORA):
        for j, generator in enumerate(GENERATORS):
            ax = axes[i, j]
            group = final[(final.corpus == corpus) & (final.generator == generator)]
            data = [group[group.chain == chain].original.to_numpy() for chain in CHAINS]
            boxes = ax.boxplot(
                data,
                tick_labels=CHAINS,
                patch_artist=True,
                showfliers=False,
                widths=0.65,
                medianprops={"color": "black"},
            )
            for box, chain in zip(boxes["boxes"], CHAINS, strict=True):
                box.set_facecolor(COLORS[chain])
                box.set_alpha(0.3)
            for k, (values, chain) in enumerate(zip(data, CHAINS, strict=True), 1):
                ax.scatter(
                    k + rng.uniform(-0.2, 0.2, len(values)),
                    values,
                    s=5,
                    alpha=0.4,
                    color=COLORS[chain],
                    linewidths=0,
                )
            ax.tick_params(axis="x", labelrotation=65, labelsize=9)
            ax.set_ylim(0, 1)
            if j == 0:
                ax.set_ylabel("Final STDI")
    save_figure(fig, "final_distributions")
    fig, axes = axes_grid()
    for i, corpus in enumerate(CORPORA):
        for j, generator in enumerate(GENERATORS):
            ax = axes[i, j]
            group = final[(final.corpus == corpus) & (final.generator == generator)]
            wide = group.pivot(index="news_id", columns="chain", values="original")
            for k, (left, right) in enumerate(CONTRASTS):
                values = wide[left] - wide[right]
                ax.scatter(
                    k + rng.uniform(-0.16, 0.16, 50),
                    values,
                    s=6,
                    alpha=0.22,
                    color="#376b93",
                    linewidths=0,
                )
                ax.plot(k, values.mean(), "D", color="#bd3e3e", ms=5)
            ax.axhline(0, color="black", lw=0.8)
            ax.set_xticks(range(6), [f"{a}-{b}" for a, b in CONTRASTS], rotation=70, fontsize=8)
            if j == 0:
                ax.set_ylabel("Paired final difference")
    save_figure(fig, "paired_contrasts")
    fig, axes = plt.subplots(2, 3, figsize=(9.4, 6.3))
    for i, corpus in enumerate(CORPORA):
        for j, generator in enumerate(GENERATORS):
            ax = axes[i, j]
            group = final[(final.corpus == corpus) & (final.generator == generator)]
            heat = group.groupby("chain")[[f"{c}_vs_original" for c in COMPONENTS]].mean()
            image = ax.imshow(heat.reindex(CHAINS), vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
            ax.set_yticks(range(8), CHAINS, fontsize=9)
            ax.set_xticks(range(6), COMPONENT_LABELS, rotation=65, ha="right", fontsize=9)
            ax.set_title(f"{corpus} | {generator}")
    fig.tight_layout(rect=(0, 0, 0.94, 1))
    color_ax = fig.add_axes((0.95, 0.28, 0.015, 0.45))
    fig.colorbar(image, cax=color_ax)
    fig.savefig(FIGURES / "components.pdf", bbox_inches="tight")
    fig.savefig(OUT / "components.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    fig, axes = axes_grid()
    for i, corpus in enumerate(CORPORA):
        for j, generator in enumerate(GENERATORS):
            ax = axes[i, j]
            group = final[(final.corpus == corpus) & (final.generator == generator)]
            for k, (dimension, color) in enumerate(
                zip(
                    ["valence", "arousal", "dominance"],
                    ["#3568a8", "#bb563d", "#668c40"],
                    strict=True,
                )
            ):
                values = group.groupby("chain")[f"llm_signed_{dimension}"].mean().reindex(CHAINS)
                ax.bar(
                    np.arange(8) + (k - 1) * 0.25,
                    values,
                    width=0.24,
                    color=color,
                    label=dimension.title(),
                )
            ax.set_xticks(range(8), CHAINS, rotation=65, fontsize=9)
            ax.axhline(0, color="black", lw=0.7)
            if j == 0:
                ax.set_ylabel("Signed LLM VAD change")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGURES / "vad_signed_changes.pdf", bbox_inches="tight")
    fig.savefig(OUT / "vad_signed_changes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    en = final[final.corpus == "English"]
    fig, axes = plt.subplots(2, 3, figsize=(9.4, 6.4), sharex=True, sharey=True)
    for j, generator in enumerate(GENERATORS):
        subset = en[en.generator == generator]
        for i, (left, right, title) in enumerate(
            [
                ("cluster_original", "llm_original", "Paired VAD estimators"),
                ("cluster_common_original", "llm_common_original", "Common Dual VAD"),
            ]
        ):
            ax = axes[i, j]
            for chain in CHAINS:
                group = subset[subset.chain == chain]
                ax.scatter(
                    group[left], group[right], color=COLORS[chain], alpha=0.42, s=9, linewidths=0
                )
            ax.plot([0, 1], [0, 1], color="black", lw=0.8)
            ax.set(xlim=(0, 1), ylim=(0, 1), title=f"{generator} | {title}")
            if i == 1:
                ax.set_xlabel("Cluster STDI")
            if j == 0:
                ax.set_ylabel("LLM STDI")
    save_figure(fig, "method_agreement", legend=True)
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 4), sharey=True)
    fig2, axes2 = plt.subplots(1, 3, figsize=(9.4, 4), sharey=True)
    for j, generator in enumerate(GENERATORS):
        subset = en[en.generator == generator]
        for offset, left, right, label, color in [
            (-0.13, "cluster_original", "llm_original", "Paired estimators", "#3e759c"),
            (0.13, "cluster_common_original", "llm_common_original", "Common Dual VAD", "#b65b43"),
        ]:
            values = (
                subset.assign(gap=subset[left] - subset[right])
                .groupby("chain")
                .gap.mean()
                .reindex(CHAINS)
            )
            axes[j].plot(np.arange(8) + offset, values, "o", ms=5, label=label, color=color)
        axes[j].axhline(0, color="black", lw=0.7)
        axes[j].set_xticks(range(8), CHAINS, rotation=65, fontsize=9)
        axes[j].set_title(generator)
        if j == 0:
            axes[j].set_ylabel("Mean Cluster - LLM STDI")
        for component, label in zip(COMPONENTS[:4], COMPONENT_LABELS[:4], strict=True):
            values = (
                subset.assign(gap=subset[f"cluster_{component}"] - subset[f"llm_{component}"])
                .groupby("chain")
                .gap.mean()
                .reindex(CHAINS)
            )
            axes2[j].plot(range(8), values, marker="o", ms=3, label=label)
        axes2[j].axhline(0, color="black", lw=0.7)
        axes2[j].set_xticks(range(8), CHAINS, rotation=65, fontsize=9)
        axes2[j].set_title(generator)
        if j == 0:
            axes2[j].set_ylabel("Mean component difference")
    for chart, ax, name in [
        (fig, axes[0], "method_differences"),
        (fig2, axes2[0], "method_components"),
    ]:
        handles, labels = ax.get_legend_handles_labels()
        chart.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
        chart.tight_layout(rect=(0, 0.09, 1, 1))
        chart.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
        chart.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(chart)
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 4.2), sharey=True)
    for j, generator in enumerate(GENERATORS):
        ax = axes[j]
        for offset, corpus, color in [
            (-0.17, "English", "#3e759c"),
            (0.17, "Portuguese", "#c66e43"),
        ]:
            subset = final[(final.corpus == corpus) & (final.generator == generator)]
            group = subset.groupby("chain").llm_original
            median = group.median().reindex(CHAINS)
            ax.errorbar(
                np.arange(8) + offset,
                median,
                yerr=[
                    median - group.quantile(0.25).reindex(CHAINS),
                    group.quantile(0.75).reindex(CHAINS) - median,
                ],
                fmt="o",
                capsize=2,
                color=color,
                label=corpus,
                ms=4,
            )
        ax.set_xticks(range(8), CHAINS, rotation=65, fontsize=9)
        ax.set_title(generator)
        ax.set_ylim(0, 0.9)
        if j == 0:
            ax.set_ylabel("Final LLM STDI: median and IQR")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIGURES / "corpus_llm_final.pdf", bbox_inches="tight")
    fig.savefig(OUT / "corpus_llm_final.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    association = tables["feature_associations"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.7), sharey=True)
    selected = ["numeric_tokens", "attribution_cues", "uncertainty_cues", "entities", "relations"]
    labels = ["Numeric tokens", "Attribution cues", "Uncertainty cues", "Entities", "Relations"]
    for i, corpus in enumerate(CORPORA):
        ax = axes[i]
        for j, generator in enumerate(GENERATORS):
            subset = association[
                (association.corpus == corpus) & (association.generator == generator)
            ].set_index("feature")
            ax.plot(
                np.arange(5) + (j - 1) * 0.12,
                subset.loc[selected, "spearman"],
                "o",
                ms=5,
                label=generator,
            )
            ax.plot(
                np.arange(5) + (j - 1) * 0.12,
                subset.loc[selected, "length_adjusted_rank_correlation"],
                "x",
                ms=5,
                color=plt.rcParams["axes.prop_cycle"].by_key()["color"][j],
            )
        ax.set_xticks(range(5), labels, rotation=40, ha="right")
        ax.set_title(corpus)
        ax.axhline(0, color="black", lw=0.7)
        ax.set_ylim(-0.8, 0.8)
        ax.set_ylabel("Rank association (dot: raw; x: length-adjusted)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIGURES / "news_features.pdf", bbox_inches="tight")
    fig.savefig(OUT / "news_features.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    fig, axes = axes_grid()
    article = tables["article_outcomes"]
    for i, corpus in enumerate(CORPORA):
        for j, generator in enumerate(GENERATORS):
            group = article[(article.corpus == corpus) & (article.generator == generator)]
            ax = axes[i, j]
            ax.scatter(group.original_words, group.outcome, color="#356c91", s=22, alpha=0.8)
            if j == 0:
                ax.set_ylabel("Mean final LLM STDI (8 paths)")
            if i == 1:
                ax.set_xlabel("Original whitespace word count")
    save_figure(fig, "length_and_drift")


def export_latex_tables(paths: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> None:
    """Write numerical tables as macros placed by the Portuguese chapter source."""
    final = paths[paths.iteration == 4]
    definitions = ["% Generated by scripts/analyze_tcc_chapter_5.py; do not edit values manually."]

    def number(value: float) -> str:
        return f"{value:.3f}".replace(".", ",")

    def table(
        command: str,
        caption: str,
        label: str,
        columns: str,
        header: str,
        rows: list[str],
        note: str,
    ) -> None:
        definitions.append(
            "\\newcommand{\\" + command + "}{%\n"
            "\\begin{table}[htbp]\n\\centering\n\\small\n"
            "\\caption{" + caption + "}\n\\label{" + label + "}\n"
            "\\begin{tabular}{"
            + columns
            + "}\n\\hline\n"
            + header
            + " \\\\\n\\hline\n"
            + " \\\\\n".join(rows)
            + " \\\\\n\\hline\n\\end{tabular}\n"
            "\\legend{Fonte: elaboração própria. " + note + "}\n\\end{table}%\n}"
        )

    header = (
        " & \\multicolumn{3}{c}{Inglês} & \\multicolumn{3}{c}{Português} \\\\\n"
        "Trajetória & GPT & Grok & Gemini & GPT & Grok & Gemini"
    )
    for metric, command, caption, label in [
        (
            "original",
            "TabelaCincoFinal",
            "STDI médio final contra o original nas configurações selecionadas.",
            "tab:resultados-final",
        ),
        (
            "incremental",
            "TabelaCincoIncremental",
            "STDI incremental médio na quarta posição.",
            "tab:resultados-incremental",
        ),
        (
            "llm_original",
            "TabelaCincoLlm",
            "STDI médio final com comparação estrutural LLM e VAD LLM nos dois corpora.",
            "tab:resultados-llm-corpora",
        ),
        (
            "llm_incremental",
            "TabelaCincoLlmIncremental",
            "STDI incremental médio na quarta posição com comparação estrutural LLM e VAD LLM.",
            "tab:resultados-llm-incremental",
        ),
    ]:
        rows = []
        for chain in CHAINS:
            values = [
                final[
                    (final.corpus == corpus)
                    & (final.generator == generator)
                    & (final.chain == chain)
                ][metric].mean()
                for corpus in CORPORA
                for generator in GENERATORS
            ]
            rows.append(chain + " & " + " & ".join(number(value) for value in values))
        note = (
            "50 entradas por célula. Inglês: Dual/Dual; português: LLM/LLM."
            if not metric.startswith("llm_")
            else "50 entradas por célula; corpora distintos, sem pareamento entre idiomas."
        )
        table(command, caption, label, "lrrrrrr", header, rows, note)
    contrast = tables["paired_contrasts"]
    rows = []
    for left, right in CONTRASTS:
        name = f"{left}-{right}"
        values = [
            contrast[
                (contrast.corpus == corpus)
                & (contrast.generator == generator)
                & (contrast.contrast == name)
                & (contrast.metric == "original")
            ]["mean"].iloc[0]
            for corpus in CORPORA
            for generator in GENERATORS
        ]
        rows.append(name + " & " + " & ".join(number(value) for value in values))
    table(
        "TabelaCincoContrastes",
        "Diferenças médias pareadas entre trajetórias na quarta posição.",
        "tab:resultados-contrastes",
        "lrrrrrr",
        header.replace("Trajetória", "Contraste"),
        rows,
        "50 diferenças por célula; valor positivo indica maior STDI na primeira trajetória.",
    )
    rows = []
    english = final[final.corpus == "English"]
    for generator in GENERATORS:
        group = english[english.generator == generator]
        for condition, left, right in [
            ("Próprio", "cluster_original", "llm_original"),
            ("Comum", "cluster_common_original", "llm_common_original"),
        ]:
            difference = group[left] - group[right]
            values = [
                group[left].mean(),
                group[right].mean(),
                difference.mean(),
                difference.abs().mean(),
                rank_correlation(group[left], group[right]),
            ]
            rows.append(
                generator
                + " & "
                + condition
                + " & "
                + " & ".join(number(value) for value in values)
            )
    table(
        "TabelaCincoMetodos",
        "Comparação dos escores finais dos métodos no corpus inglês.",
        "tab:resultados-metodos",
        "llrrrrr",
        "Gerador & VAD & Cluster & LLM & Dif. & DMA & $\\rho_s$",
        rows,
        "400 posições finais por gerador (50 notícias em oito trajetórias), sem independência "
        "entre trajetórias. Próprio: BERT no Cluster e LLM no comparador LLM; "
        "comum: VAD Dual nos dois. Dif.: média de Cluster menos LLM; "
        "DMA: diferença absoluta média; $\\rho_s$: correlação de Spearman.",
    )
    features = tables["feature_associations"]
    rows = []
    for feature, label in [
        ("original_words", "Palavras"),
        ("numeric_tokens", "Números"),
        ("attribution_cues", "Atribuição"),
        ("uncertainty_cues", "Incerteza"),
        ("entities", "Entidades"),
        ("relations", "Relações"),
    ]:
        values = [
            features[
                (features.corpus == corpus)
                & (features.generator == generator)
                & (features.feature == feature)
            ].spearman.iloc[0]
            for corpus in CORPORA
            for generator in GENERATORS
        ]
        rows.append(label + " & " + " & ".join(number(value) for value in values))
    table(
        "TabelaCincoAtributos",
        "Associações de Spearman entre características originais e deriva final por LLM.",
        "tab:resultados-atributos",
        "lrrrrrr",
        header.replace("Trajetória", "Característica"),
        rows,
        "50 notícias por corpus e gerador. O desfecho é a média dos oito STDIs finais LLM/LLM "
        "de cada notícia; números, atribuição e incerteza são indicadores lexicais exploratórios.",
    )
    target = ROOT / "tcc/elementos-textuais/capitulo-5-tabelas.tex"
    target.write_text("\n\n".join(definitions) + "\n", encoding="utf-8")


def export_case_candidates(paths: pd.DataFrame) -> None:
    final = paths[paths.iteration == 4].copy()
    median = final.groupby(["corpus", "generator", "chain"]).original.transform("median")
    final["distance_to_group_median"] = (final.original - median).abs()
    candidates = []
    for corpus in CORPORA:
        group = final[final.corpus == corpus]
        for reason, selection in [
            ("typical", group[group.chain == "CCPP"].nsmallest(2, "distance_to_group_median")),
            ("largest_drift", group.nlargest(2, "original")),
            ("neutral_stabilization", group[group.chain == "DDNN"].nlargest(2, "original")),
            ("return_toward_original", group[group.chain != "NNNN"].nsmallest(2, "delta_original")),
        ]:
            for _, row in selection.iterrows():
                candidates.append({"selection_reason": reason} | row.to_dict())
    english = final[final.corpus == "English"].copy()
    english["method_gap"] = english.cluster_common_original - english.llm_common_original
    for _, row in english.loc[english.method_gap.abs().nlargest(4).index].iterrows():
        candidates.append({"selection_reason": "method_disagreement"} | row.to_dict())
    details = []
    for candidate in candidates:
        target = paths[
            (paths.run == candidate["run"])
            & (paths.news_id == candidate["news_id"])
            & (paths.chain == candidate["chain"])
        ].sort_values("iteration")
        versions = []
        for _, row in target.iterrows():
            with (ROOT / row.source_file).open(encoding="utf-8") as stream:
                for number, line in enumerate(stream, 1):
                    if number == row.source_line:
                        raw = json.loads(line)
                        break
            versions.append(
                {
                    "iteration": int(row.iteration),
                    "text": raw["rewritten_text"],
                    "original": row.original,
                    "incremental": row.incremental,
                    "original_text": raw["metadata_original_text"],
                    "title": raw.get("metadata_title", ""),
                    "source_file": row.source_file,
                    "source_line": row.source_line,
                }
            )
        details.append(
            {
                "selection_reason": candidate["selection_reason"],
                "corpus": candidate["corpus"],
                "generator": candidate["generator"],
                "chain": candidate["chain"],
                "news_id": candidate["news_id"],
                "versions": versions,
            }
        )
    (OUT / "case_candidates.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    nodes, features, sources = load_data()
    paths = project_paths(nodes)
    tables = summarize(paths, features)
    for name, frame in {
        "unique_nodes": nodes,
        "path_positions": paths,
        "original_features": features,
        **tables,
    }.items():
        write_csv(frame, name)
    create_figures(paths, tables)
    export_latex_tables(paths, tables)
    export_case_candidates(paths)
    corpora = [
        ROOT / "data/graphs/graph_politics_news.csv",
        ROOT / "data/elections_2026/simulation_news_100_cleaned.csv",
    ]
    manifest = {
        "source_files": sources
        + [{"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path)} for path in corpora],
        "unique_nodes": len(nodes),
        "projected_positions": len(paths),
        "original_texts": len(features),
        "new_model_calls": 0,
        "analysis": "Descriptive, paired within corpus; no hypothesis tests or population CIs",
        "features": "Bilingual lexical proxies, frozen original text; first GPT/N-tree extraction",
        "feature_outcome": "Mean of eight final LLM/LLM path scores per article and generator",
        "feature_rules": {
            "numbers": NUMBER.pattern,
            "attribution": ATTRIBUTION.pattern,
            "uncertainty": UNCERTAINTY.pattern,
        },
        "versions": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "script_sha256": digest(Path(__file__)),
        "figures": [p.relative_to(ROOT).as_posix() for p in sorted(FIGURES.glob("*.pdf"))],
        "checks": [
            "24 source files, 1200 unique nodes per run, 50 inputs per run",
            "Input IDs and trimmed bodies match the first 50 eligible CSV rows",
            "No duplicated node keys; original text identical within corpus/article",
            "Selected STDI and incremental valid in every node",
            "All shared path prefixes identical; four positions in every path",
            "Complete cumulative score equals sum of four increments",
            "Dual score equals mean of branches evaluated with common Dual VAD",
            "Portuguese selected scores equal LLM branch scores",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "unique_nodes": len(nodes),
                "path_positions": len(paths),
                "figures": len(manifest["figures"]),
                "output": str(OUT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
