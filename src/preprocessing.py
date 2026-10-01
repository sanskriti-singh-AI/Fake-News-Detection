from pathlib import Path

import pandas as pd


LABEL_MAP = {
    "Fake": 0,
    "True": 1,
}


def load_news_data(data_path: str | Path) -> pd.DataFrame:
    """
    Load the processed news dataset used by the project.

    Parameters
    ----------
    data_path : str or Path
        Path to the processed modelling dataset.

    Returns
    -------
    pd.DataFrame
        Loaded news dataset.
    """
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {data_path}"
        )

    data = pd.read_csv(data_path)

    required_columns = [
        "title",
        "text",
        "subject",
        "date",
        "label",
        "label_name",
        "model_text",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return data


def prepare_text_data(
    data: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """
    Prepare the text feature and target labels.

    The primary model input is the combined title and article body
    stored in the model_text column.
    """
    text = data["model_text"].fillna("").astype(str)
    target = data["label"].astype(int)

    if text.str.strip().eq("").any():
        raise ValueError(
            "Empty text documents were found in model_text."
        )

    if not target.isin([0, 1]).all():
        raise ValueError(
            "Target labels must contain only 0 and 1."
        )

    return text, target