from pathlib import Path

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "fake_news_final_model.joblib"
)


LABEL_MAP = {
    0: "Fake",
    1: "True",
}


def load_model():
    """Load the trained fake-news classification pipeline."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Final model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def predict_article(title: str, text: str) -> dict:
    """Predict the dataset label for a single news article."""

    if not isinstance(title, str):
        raise TypeError("Title must be a string.")

    if not isinstance(text, str):
        raise TypeError("Text must be a string.")

    title = title.strip()
    text = text.strip()

    if not title and not text:
        raise ValueError(
            "Title or article text must contain content."
        )

    model = load_model()

    combined_text = f"{title} {text}".strip()

    prediction = int(
        model.predict([combined_text])[0]
    )

    result = {
        "prediction": prediction,
        "label": LABEL_MAP[prediction],
    }

    if hasattr(model, "decision_function"):
        decision_score = float(
            model.decision_function([combined_text])[0]
        )

        result["decision_score"] = decision_score

    return result


if __name__ == "__main__":
    example_result = predict_article(
        title="Government announces new economic policy",
        text=(
            "The government announced a new economic policy "
            "during a press conference today."
        ),
    )

    print("\nFake News Prediction")
    print("-" * 30)
    print(f"Predicted label: {example_result['label']}")
    print(f"Decision score: {example_result['decision_score']:.4f}")