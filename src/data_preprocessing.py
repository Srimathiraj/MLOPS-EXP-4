from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import joblib
import json

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def preprocess_data():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
    )
    joblib.dump((X_train, X_test, y_train, y_test), DATA_DIR / "dataset.joblib")
    metadata = {
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "target_names": iris.target_names.tolist()
    }
    (DATA_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Data preprocessed successfully. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

if __name__ == "__main__":
    preprocess_data()
