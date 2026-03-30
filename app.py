from __future__ import annotations

from math import isfinite
from pathlib import Path
import os
from typing import Any

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS


FEATURES = ["food", "transport", "shopping"]


class APIError(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class ModelState:
    def __init__(
        self,
        model: Any | None,
        metadata: dict[str, Any],
        model_path: Path | None,
        load_error: str | None,
    ) -> None:
        self.model = model
        self.metadata = metadata
        self.model_path = model_path
        self.load_error = load_error


class ModelManager:
    def __init__(self) -> None:
        self.candidate_paths = [
            Path(os.getenv("MODEL_PATH", "")).expanduser() if os.getenv("MODEL_PATH") else None,
            Path(__file__).resolve().parent / "expense_model.pkl",
            Path(__file__).resolve().parents[1] / "ml-model" / "expense_model.pkl",
        ]
        self.state = self._load_state()

    def _load_state(self) -> ModelState:
        try:
            candidates = [p for p in self.candidate_paths if p and p.exists()]
            if not candidates:
                searched = ", ".join(str(p) for p in self.candidate_paths if p)
                return ModelState(None, {}, None, f"Model file not found. Searched: {searched}")

            model_path = max(candidates, key=lambda p: p.stat().st_mtime)
            loaded = joblib.load(model_path)

            if isinstance(loaded, dict) and "model" in loaded:
                metadata = loaded.get("metadata") or {}
                return ModelState(loaded["model"], metadata, model_path, None)

            return ModelState(loaded, {}, model_path, None)
        except Exception as exc:
            return ModelState(None, {}, None, f"Failed to load model: {str(exc)}")

    def ensure_available(self) -> None:
        if self.state.model is None:
            raise APIError(self.state.load_error or "Model unavailable", 503)


class InputValidator:
    @staticmethod
    def parse_payload(payload: Any) -> dict[str, float]:
        if not isinstance(payload, dict):
            raise APIError("Request body must be a valid JSON object", 400)

        parsed: dict[str, float] = {}
        for field in FEATURES:
            if field not in payload:
                raise APIError(f"Missing required field: {field}", 400)

            parsed[field] = InputValidator._parse_number(payload[field], field)

        budget_value = payload.get("budget")
        if budget_value is not None:
            parsed["budget"] = InputValidator._parse_number(budget_value, "budget")

        return parsed

    @staticmethod
    def _parse_number(value: Any, field_name: str) -> float:
        if value is None:
            raise APIError(f"{field_name} cannot be null", 400)

        try:
            if isinstance(value, str):
                cleaned = value.strip().replace(",", "").replace("₹", "").replace("$", "")
            else:
                cleaned = value
            parsed = float(cleaned)
        except (TypeError, ValueError):
            raise APIError(f"{field_name} must be a numeric value", 400)

        if not isfinite(parsed):
            raise APIError(f"{field_name} must be a finite number", 400)
        if parsed < 0:
            raise APIError(f"{field_name} must be non-negative", 400)

        return parsed


class PredictionEngine:
    def __init__(self, model_state: ModelState) -> None:
        self.model_state = model_state
        self.metadata = model_state.metadata if isinstance(model_state.metadata, dict) else {}

    def predict(self, inputs: dict[str, float]) -> dict[str, Any]:
        food = inputs["food"]
        transport = inputs["transport"]
        shopping = inputs["shopping"]
        budget = inputs.get("budget")

        actual_total = food + transport + shopping
        ml_prediction = self._predict_ml(food, transport, shopping)
        final_prediction = 0.6 * ml_prediction + 0.4 * actual_total
        final_prediction = max(0.0, final_prediction)
        
        variance = abs(ml_prediction - actual_total) / max(actual_total, 1.0)
        confidence_score = self._compute_confidence(variance)
        confidence = self._confidence_label(variance)
        
        expense_status = self._expense_status(final_prediction, budget)
        remaining_budget = None if budget is None else max(0.0, budget - final_prediction)
        suggestion = self._suggestion(food, transport, shopping)
        breakdown = self._breakdown(food, transport, shopping)

        return {
            "status": "success",
            "predicted_expense": round(final_prediction, 2),
            "actual_total": round(actual_total, 2),
            "ml_prediction": round(ml_prediction, 2),
            "remaining_budget": round(remaining_budget, 2) if remaining_budget is not None else None,
            "confidence": confidence,
            "confidence_score": round(confidence_score, 3),
            "expense_status": expense_status,
            "suggestion": suggestion,
            "breakdown": breakdown,
        }

    def _predict_ml(self, food: float, transport: float, shopping: float) -> float:
        model = self.model_state.model
        if model is None:
            raise APIError("Model unavailable", 503)

        try:
            features = self.metadata.get("features", FEATURES)
            row = {
                "food": food,
                "transport": transport,
                "shopping": shopping,
            }
            model_input = pd.DataFrame([[row.get(name, 0.0) for name in features]], columns=features)
            predicted = float(model.predict(model_input)[0])
            return max(0.0, predicted)
        except APIError:
            raise
        except Exception as exc:
            raise APIError(f"Model prediction failed: {str(exc)}", 500)

    @staticmethod
    def _compute_confidence(variance: float) -> float:
        if variance < 0.2:
            return 0.9
        if variance < 0.5:
            return 0.6
        return 0.3

    @staticmethod
    def _confidence_label(variance: float) -> str:
        if variance < 0.2:
            return "high"
        if variance < 0.5:
            return "medium"
        return "low"

    @staticmethod
    def _expense_status(predicted_expense: float, budget: float | None) -> str:
        if budget is None:
            return "Not Available"

        if budget <= 0:
            return "High"

        ratio = predicted_expense / budget
        if ratio > 0.8:
            return "High"
        if ratio > 0.5:
            return "Moderate"
        return "Low"

    @staticmethod
    def _suggestion(food: float, transport: float, shopping: float) -> str:
        categories = {
            "food": food,
            "transport": transport,
            "shopping": shopping,
        }
        highest = max(categories, key=categories.get)

        if highest == "shopping":
            return "Shopping is your highest expense. Try reducing non-essential purchases."
        if highest == "food":
            return "Food is your highest expense. Consider meal planning to save more."
        return "Transport is your highest expense. Look for commute optimization opportunities."

    @staticmethod
    def _breakdown(food: float, transport: float, shopping: float) -> dict[str, float]:
        total = food + transport + shopping
        if total <= 0:
            return {
                "food": 0.0,
                "transport": 0.0,
                "shopping": 0.0,
            }

        return {
            "food": round((food / total) * 100.0, 2),
            "transport": round((transport / total) * 100.0, 2),
            "shopping": round((shopping / total) * 100.0, 2),
        }


app = Flask(__name__)
CORS(app)
model_manager = ModelManager()


@app.get("/")
def health() -> Any:
    state = model_manager.state
    return jsonify(
        {
            "message": "Finance expense prediction API is running",
            "model_loaded": state.model is not None,
            "model_file": state.model_path.name if state.model_path else None,
            "features": state.metadata.get("features", FEATURES),
        }
    )


@app.post("/predict")
def predict() -> Any:
    try:
        model_manager.ensure_available()
        payload = request.get_json(silent=True)
        inputs = InputValidator.parse_payload(payload)
        engine = PredictionEngine(model_manager.state)
        result = engine.predict(inputs)
        return jsonify(result), 200
    except APIError as exc:
        return jsonify({"status": "error", "message": exc.message}), exc.status_code
    except Exception:
        return jsonify({"status": "error", "message": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
