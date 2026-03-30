from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

FEATURES = ["food", "transport", "shopping"]
TARGET = "total_expense"
MODEL_PATH = Path(__file__).resolve().parent / "expense_model.pkl"
RANDOM_STATE = 42


def build_synthetic_dataset(samples: int = 1200) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)

    food = rng.uniform(100, 1000, samples)
    transport = rng.uniform(50, 800, samples)
    shopping = rng.uniform(100, 5000, samples)
    noise = rng.normal(0, 120, samples)

    total_expense = 0.85 * food + 0.9 * transport + 0.95 * shopping + noise
    total_expense = np.clip(total_expense, 0, None)

    return pd.DataFrame(
        {
            "food": food,
            "transport": transport,
            "shopping": shopping,
            TARGET: total_expense,
        }
    )


def train_and_save() -> None:
    df = build_synthetic_dataset().dropna().copy()

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    candidates = {
        "LinearRegression": LinearRegression(),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.2, random_state=RANDOM_STATE, max_iter=5000),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }

    best_name = None
    best_model = None
    best_cv_mae = float("inf")

    for name, model in candidates.items():
        cv_mae = -cross_val_score(
            model, X_train, y_train, cv=5, scoring="neg_mean_absolute_error"
        ).mean()

        if cv_mae < best_cv_mae:
            best_cv_mae = cv_mae
            best_name = name
            best_model = model

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    metrics = {
        "r2": float(r2_score(y_test, y_pred)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "cv_mae": float(best_cv_mae),
    }

    feature_ranges = {
        feature: (float(df[feature].min()), float(df[feature].max())) for feature in FEATURES
    }

    payload = {
        "model": best_model,
        "metadata": {
            "features": FEATURES,
            "target": TARGET,
            "feature_ranges": feature_ranges,
            "model_name": best_name,
            "metrics": metrics,
            "row_count": int(len(df)),
        },
    }

    joblib.dump(payload, MODEL_PATH)

    print(f"Saved model: {MODEL_PATH}")
    print(f"Selected model: {best_name}")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    train_and_save()
