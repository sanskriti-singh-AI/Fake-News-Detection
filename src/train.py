from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from preprocessing import (
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

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "fake_news_final_model.joblib"
)


def build_model() -> Pipeline:
    """Build the final TF-IDF + LinearSVC pipeline."""

    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                    max_features=100000,
                ),
            ),
            (
                "model",
                LinearSVC(
                    C=2.0,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def train_model() -> None:
    """Train the final model on the processed dataset."""

    data = load_news_data(DATA_PATH)
    text, target = prepare_text_data(data)

    model = build_model()
    model.fit(text, target)

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    print("Final model trained successfully.")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()