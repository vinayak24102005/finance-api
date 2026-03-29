from pathlib import Path
from math import isfinite

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

DEFAULT_FEATURES = ["food", "transport", "shopping"]

# Fallback ranges in case model metadata is missing.
DEFAULT_FEATURE_RANGES = {
    "food": (80.0, 450.0),
    "transport": (20.0, 220.0),
    "shopping": (40.0, 500.0),
}

MODEL_CANDIDATE_PATHS = [
    Path(__file__).resolve().parent / "expense_model.pkl",
    Path(__file__).resolve().parents[1] / "ml-model" / "expense_model.pkl",
]


def load_model():
    existing_candidates = [path for path in MODEL_CANDIDATE_PATHS if path.exists()]
    if existing_candidates:
        # Prefer newest artifact if both model files exist.
        model_path = max(existing_candidates, key=lambda p: p.stat().st_mtime)
        loaded = joblib.load(model_path)

        if isinstance(loaded, dict) and "model" in loaded:
            return loaded["model"], loaded.get("metadata") or {}, model_path

        return loaded, {}, model_path

    searched_paths = ", ".join(str(path) for path in MODEL_CANDIDATE_PATHS)
    raise FileNotFoundError(f"Model file not found. Searched: {searched_paths}")


model, model_metadata, model_path = load_model()
FEATURES = model_metadata.get("features", DEFAULT_FEATURES)


def parse_feature(data: dict, field_name: str) -> float:
    if field_name not in data:
        raise KeyError(field_name)

    value = data[field_name]
    if value is None:
        raise ValueError(f"'{field_name}' cannot be null")

    if isinstance(value, str):
        normalized = value.strip().replace(",", "")
        if normalized.startswith("$"):
            normalized = normalized[1:]
    else:
        normalized = value

    try:
        parsed_value = float(normalized)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{field_name}' must be a valid number") from exc

    if not isfinite(parsed_value):
        raise ValueError(f"'{field_name}' must be a finite number")
    if parsed_value < 0:
        raise ValueError(f"'{field_name}' must be greater than or equal to 0")

    return parsed_value


def get_feature_ranges() -> dict:
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


def clamp_to_range(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def get_model_quality_score() -> float:
    metrics = model_metadata.get("metrics", {}) if isinstance(model_metadata, dict) else {}
    r2 = metrics.get("r2", model_metadata.get("r2")) if isinstance(metrics, dict) else model_metadata.get("r2")
    if isinstance(r2, (int, float)):
        return max(0.0, min(1.0, float(r2)))
    return 0.75


def get_input_stability_score(adjustment_count: int, total_features: int) -> float:
    if total_features <= 0:
        return 0.5
    ratio = adjustment_count / total_features
    return max(0.0, 1.0 - ratio)


def to_confidence_label(score: float) -> str:
    if score >= 0.85:
        return "high"
    if score >= 0.65:
        return "medium"
    return "low"


@app.get("/")
def health_check():
    return jsonify(
        {
            "message": "Expense prediction API is running",
            "model_file": str(model_path.name),
            "features": FEATURES,
            "model_name": model_metadata.get("model_name", type(model).__name__),
            "model_metrics": model_metadata.get("metrics", {}),
        }
    )


@app.post("/predict")
def predict_expense():
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({
                "status": "error",
                "message": "Request body must be a valid JSON object",
            }), 400

        raw_inputs = {feature: parse_feature(data, feature) for feature in FEATURES}
        feature_ranges = get_feature_ranges()

        adjusted_inputs = {}
        adjustments = []
        out_of_distribution = False

        for feature in FEATURES:
            raw_value = raw_inputs[feature]
            low, high = feature_ranges[feature]
            clipped_value = clamp_to_range(raw_value, low, high)
            adjusted_inputs[feature] = clipped_value

            if raw_value != clipped_value:
                out_of_distribution = True
                adjustments.append(
                    {
                        "feature": feature,
                        "input": raw_value,
                        "used_for_prediction": clipped_value,
                        "allowed_range": [low, high],
                    }
                )

        model_input = pd.DataFrame([adjusted_inputs], columns=FEATURES)
        prediction = float(model.predict(model_input)[0])
        prediction = max(0.0, prediction)

        model_quality_score = get_model_quality_score()
        stability_score = get_input_stability_score(len(adjustments), len(FEATURES))
        confidence_score = round(0.6 * model_quality_score + 0.4 * stability_score, 3)
        confidence = to_confidence_label(confidence_score)

        return jsonify({
            "status": "success",
            "predicted_expense": round(prediction, 2),
            "confidence": confidence,
            "confidence_score": confidence_score,
            "model_input": adjusted_inputs,
            "input_adjustments": adjustments,
            "model_name": model_metadata.get("model_name", type(model).__name__),
            "model_metrics": model_metadata.get("metrics", {}),
            "note": (
                "Some inputs were outside trained ranges and were clipped for better stability"
                if out_of_distribution
                else "Inputs are within trained ranges"
            ),
        })

    except KeyError as exc:
        return jsonify({
            "status": "error",
            "message": f"Missing required field: {exc.args[0]}",
        }), 400

    except ValueError as exc:
        return jsonify({
            "status": "error",
            "message": str(exc),
        }), 400

    except Exception as exc:
        return jsonify({
            "status": "error",
            "message": f"Prediction failed: {str(exc)}",
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)