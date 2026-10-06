from pathlib import Path
import json
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
THRESHOLD = 0.85

def evaluate_model():
    X_train, X_test, y_train, y_test = joblib.load(DATA_DIR / "dataset.joblib")
    model = joblib.load(MODEL_DIR / "model.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")

    predictions = model.predict(scaler.transform(X_test))

    metrics = {
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        "precision": round(precision_score(y_test, predictions, average="weighted"), 4),
        "recall": round(recall_score(y_test, predictions, average="weighted"), 4),
        "f1_score": round(f1_score(y_test, predictions, average="weighted"), 4),
        "accuracy_threshold": THRESHOLD
    }
    metrics["passed_quality_gate"] = metrics["accuracy"] >= THRESHOLD

    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("Model Evaluation Metrics:")
    print(f"- Accuracy: {metrics['accuracy']:.4f} (Threshold: {THRESHOLD})")
    print(f"- Quality Gate Status: {'PASSED' if metrics['passed_quality_gate'] else 'FAILED'}")

    if not metrics["passed_quality_gate"]:
        raise RuntimeError("Quality gate failed: accuracy is below 0.85")

if __name__ == "__main__":
    evaluate_model()
