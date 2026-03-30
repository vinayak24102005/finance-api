from pathlib import Path
from math import isfinite
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

DEFAULT_FEATURES = ["food", "transport", "shopping"]

DEFAULT_FEATURE_RANGES = {
    "food": (100.0, 1000.0),
    "transport": (50.0, 800.0),
    "shopping": (100.0, 5000.0),
}

MODEL_CANDIDATE_PATHS = [
    Path(os.getenv("MODEL_PATH", "")).expanduser() if os.getenv("MODEL_PATH") else None,
    Path(__file__).resolve().parent / "expense_model.pkl",
    Path(__file__).resolve().parents[1] / "finance-api" / "expense_model.pkl",
]


def load_model():
    existing_candidates = [path for path in MODEL_CANDIDATE_PATHS if path and path.exists()]
    if existing_candidates:
        model_path = max(existing_candidates, key=lambda p: p.stat().st_mtime)
        loaded = joblib.load(model_path)

        if isinstance(loaded, dict) and "model" in loaded:
            return loaded["model"], loaded.get("metadata") or {}, model_path

        return loaded, {}, model_path

    searched_paths = ", ".join(str(path) for path in MODEL_CANDIDATE_PATHS if path)
    raise FileNotFoundError(f"Model file not found. Searched: {searched_paths}")


model, model_metadata, model_path = load_model()
FEATURES = model_metadata.get("features", DEFAULT_FEATURES)


def parse_feature(data: dict, field_name: str) -> float:
    if field_name not in data:
        raise KeyError(field_name)

    value = data[field_name]

    if value is None:
        raise ValueError(f"{field_name} cannot be null")

    try:
        parsed_value = float(str(value).replace(",", "").replace("₹", ""))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number") from exc

    if not isfinite(parsed_value) or parsed_value < 0:
        raise ValueError(f"{field_name} must be greater than or equal to 0")

    return parsed_value


def get_feature_ranges():
    metadata_ranges = model_metadata.get("feature_ranges") if isinstance(model_metadata, dict) else None

    if isinstance(metadata_ranges, dict):
        merged = {}
        for feature in FEATURES:
            low_high = metadata_ranges.get(feature)
            if (
                isinstance(low_high, (list, tuple))
                and len(low_high) == 2
                and isinstance(low_high[0], (int, float))
                and isinstance(low_high[1], (int, float))
            ):
                merged[feature] = (float(low_high[0]), float(low_high[1]))
            else:
                merged[feature] = DEFAULT_FEATURE_RANGES.get(feature, (0.0, float("inf")))
        return merged

    return {feature: DEFAULT_FEATURE_RANGES.get(feature, (0.0, float("inf"))) for feature in FEATURES}


def clamp(value, low, high):
    return min(max(value, low), high)


def get_model_quality():
    metrics = model_metadata.get("metrics", {}) if isinstance(model_metadata, dict) else {}
    r2 = metrics.get("r2", model_metadata.get("r2")) if isinstance(metrics, dict) else model_metadata.get("r2")

    if isinstance(r2, (int, float)):
        return max(0.0, min(1.0, float(r2)))

    return 0.75


def get_stability(adjustments, total):
    if total <= 0:
        return 0.5
    return max(0.0, 1 - (adjustments / total))


def confidence_label(score):
    if score >= 0.85:
        return "high"
    if score >= 0.65:
        return "medium"
    return "low"


@app.get("/")
def health():
    return jsonify(
        {
            "message": "API running",
            "model": str(model_path.name),
            "features": FEATURES,
            "model_name": model_metadata.get("model_name", type(model).__name__),
            "model_metrics": model_metadata.get("metrics", {}),
        }
    )


@app.post("/predict")
def predict():
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({"status": "error", "message": "Invalid JSON"}), 400

        inputs = {f: parse_feature(data, f) for f in FEATURES}
        ranges = get_feature_ranges()

        adjusted = {}
        adjustments = []

        for f in FEATURES:
            raw = inputs[f]
            low, high = ranges[f]
            clipped = clamp(raw, low, high)
            adjusted[f] = clipped

            if raw != clipped:
                adjustments.append(f)

        model_input = pd.DataFrame([adjusted], columns=FEATURES)
        prediction = float(model.predict(model_input)[0])
        prediction = max(0.0, min(prediction, 100000.0))

        model_q = get_model_quality()
        stability = get_stability(len(adjustments), len(FEATURES))

        confidence_score = round(max(0.0, min(1.0, 0.6 * model_q + 0.4 * stability)), 3)
        confidence = confidence_label(confidence_score)

        return jsonify(
            {
                "status": "success",
                "predicted_expense": round(prediction, 2),
                "confidence": confidence,
                "confidence_score": confidence_score,
                "note": "Inputs adjusted" if adjustments else "Inputs normal",
            }
        )

    except KeyError as e:
        return jsonify({"status": "error", "message": f"Missing: {e.args[0]}"}), 400

    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 400

    except Exception:
        return jsonify({"status": "error", "message": "Internal server error"}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)