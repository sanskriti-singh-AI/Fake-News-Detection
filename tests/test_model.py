from pathlib import Path

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "fake_news_final_model.joblib"
)


def test_final_model_exists():
    assert MODEL_PATH.exists(), (
        f"Final model not found: {MODEL_PATH}"
    )


def test_final_model_loads():
    model = joblib.load(MODEL_PATH)

    assert model is not None


def test_final_model_has_prediction_interface():
    model = joblib.load(MODEL_PATH)

    assert hasattr(model, "predict")


def test_final_model_can_predict():
    model = joblib.load(MODEL_PATH)

    sample_text = [
        (
            "Government announces new policy",
            "The government announced a new policy today."
        )
    ]

    combined_text = [
        f"{title} {text}".strip()
        for title, text in sample_text
    ]

    prediction = model.predict(combined_text)

    assert len(prediction) == 1
    assert prediction[0] in [0, 1]