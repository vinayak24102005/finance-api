from pathlib import Path
import joblib
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODEL_PATH = Path(__file__).resolve().parent / "expense_model.pkl"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


model = load_model()


def parse_feature(data: dict, field_name: str) -> float:
    """Validate and parse numeric feature values from request payload."""
    if field_name not in data:
        raise KeyError(field_name)

    value = data[field_name]
    if value is None:
        raise ValueError(f"'{field_name}' cannot be null")

    try:
        parsed_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{field_name}' must be a valid number") from exc

    if parsed_value < 0:
        raise ValueError(f"'{field_name}' must be greater than or equal to 0")

    return parsed_value


@app.get("/")
def health_check():
    return jsonify({"message": "Expense prediction API is running"})


@app.post("/predict")
def predict_expense():
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({
                "status": "error",
                "message": "Request body must be a valid JSON object"
            }), 400

        food = parse_feature(data, "food")
        transport = parse_feature(data, "transport")
        shopping = parse_feature(data, "shopping")

        prediction = float(model.predict([[food, transport, shopping]])[0])

        return jsonify({
            "status": "success",
            "predicted_expense": round(prediction, 2)
        })

    except KeyError as e:
        return jsonify({
            "status": "error",
            "message": f"Missing required field: {e.args[0]}"
        }), 400

    except ValueError as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Prediction failed: {str(e)}"
        }), 500


if __name__ == "__main__":
    app.run()