from pathlib import Path

import pandas as pd

from src.preprocessing import (
    load_news_data,
    prepare_text_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "news_modeling_data.csv"
)


def test_processed_dataset_exists():
    assert DATA_PATH.exists(), (
        f"Processed dataset not found: {DATA_PATH}"
    )


def test_load_news_data():
    data = load_news_data(DATA_PATH)

    assert isinstance(data, pd.DataFrame)
    assert not data.empty

    required_columns = {
        "title",
        "text",
        "subject",
        "date",
        "label",
        "label_name",
        "model_text",
    }

    assert required_columns.issubset(data.columns)


def test_prepare_text_data():
    data = load_news_data(DATA_PATH)

    text, target = prepare_text_data(data)

    assert len(text) == len(target)
    assert len(text) > 0

    assert text.notna().all()
    assert target.notna().all()

    assert set(target.unique()).issubset({0, 1})


def test_model_text_is_not_empty():
    data = load_news_data(DATA_PATH)

    text, _ = prepare_text_data(data)

    assert text.str.strip().ne("").all()