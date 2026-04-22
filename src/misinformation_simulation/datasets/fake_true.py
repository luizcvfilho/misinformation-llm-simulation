from __future__ import annotations

from pathlib import Path

import pandas as pd

FAKE_TRUE_VAD_DATASET_NAME = "FakeVsTrueVAD"
FAKE_LABEL = "fake"
TRUE_LABEL = "true"
DEFAULT_FAKE_TRUE_VAD_DIR = (
    Path(__file__).resolve().parents[3] / "data" / FAKE_TRUE_VAD_DATASET_NAME
)

_DEFAULT_TITLE_COLUMN = "title"
_DEFAULT_TEXT_COLUMN = "text"
_LABEL_TO_FILENAME = {
    FAKE_LABEL: "Fake.csv",
    TRUE_LABEL: "True.csv",
}


def build_article_text(
    row: pd.Series,
    *,
    title_column: str = _DEFAULT_TITLE_COLUMN,
    text_column: str = _DEFAULT_TEXT_COLUMN,
) -> str:
    title = _normalize_text_fragment(row.get(title_column))
    text = _normalize_text_fragment(row.get(text_column))
    parts = [part for part in (title, text) if part]
    return "\n\n".join(parts)


def load_fake_true_news_dataset(
    data_dir: Path | str = DEFAULT_FAKE_TRUE_VAD_DIR,
    *,
    title_column: str = _DEFAULT_TITLE_COLUMN,
    text_column: str = _DEFAULT_TEXT_COLUMN,
    output_text_column: str = "article_text",
    max_rows_per_label: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    dataset_dir = Path(data_dir)
    frames = [
        _load_labeled_frame(
            dataset_dir / filename,
            label=label,
            title_column=title_column,
            text_column=text_column,
            output_text_column=output_text_column,
            max_rows=max_rows_per_label,
            random_state=random_state,
        )
        for label, filename in _LABEL_TO_FILENAME.items()
    ]
    combined = pd.concat(frames, ignore_index=True)
    combined.insert(0, "dataset_name", FAKE_TRUE_VAD_DATASET_NAME)
    return combined


def _load_labeled_frame(
    csv_path: Path,
    *,
    label: str,
    title_column: str,
    text_column: str,
    output_text_column: str,
    max_rows: int | None,
    random_state: int,
) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    _validate_expected_columns(frame, csv_path, title_column=title_column, text_column=text_column)

    if max_rows is not None and max_rows < len(frame):
        frame = frame.sample(n=max_rows, random_state=random_state).sort_index()

    frame = frame.copy()
    frame.insert(0, "source_row_index", frame.index.astype(int))
    frame.insert(1, "source_file", csv_path.name)
    frame.insert(2, "label", label)
    frame[output_text_column] = frame.apply(
        build_article_text,
        axis=1,
        title_column=title_column,
        text_column=text_column,
    )
    frame["article_char_count"] = frame[output_text_column].str.len()
    frame["article_word_count"] = frame[output_text_column].str.split().map(len).astype(int)
    return frame.reset_index(drop=True)


def _normalize_text_fragment(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    return text if text and text.lower() != "nan" else ""


def _validate_expected_columns(
    frame: pd.DataFrame,
    csv_path: Path,
    *,
    title_column: str,
    text_column: str,
) -> None:
    missing_columns = [
        column for column in (title_column, text_column) if column not in frame.columns
    ]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns in {csv_path.name}: {missing}")
