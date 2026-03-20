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


@app.get("/")
def health_check():
    return jsonify({"message": "Expense prediction API is running"})


@app.post("/predict")
def predict_expense():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    required_fields = ["food", "transport", "shopping"]
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400

    try:
        food = float(data["food"])
        transport = float(data["transport"])
        shopping = float(data["shopping"])

        prediction = float(model.predict([[food, transport, shopping]])[0])

        return jsonify({
            "status": "success",
            "predicted_expense": round(prediction, 2)
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


if __name__ == "__main__":
    app.run()