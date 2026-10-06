from pathlib import Path
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROD_DIR = ROOT / "production_model"

def predict():
    model = joblib.load(PROD_DIR / "model.joblib")
    scaler = joblib.load(PROD_DIR / "scaler.joblib")

    sample = np.array([[5.1, 3.5, 1.4, 0.2]])
    probabilities = model.predict_proba(scaler.transform(sample))[0]
    class_id = int(model.predict(scaler.transform(sample))[0])

    result = [{
        "class_id": class_id,
        "probabilities": [round(float(x), 4) for x in probabilities]
    }]
    print(f"Sample Input: {sample.tolist()}")
    print(f"Prediction Output: {result}")
    return result

if __name__ == "__main__":
    predict()
