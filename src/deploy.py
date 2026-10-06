from pathlib import Path
import json
import shutil
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
PROD_DIR = ROOT / "production_model"
PROD_DIR.mkdir(exist_ok=True)

def deploy_model():
    metrics = json.loads((MODEL_DIR / "metrics.json").read_text())
    if not metrics["passed_quality_gate"]:
        raise RuntimeError("Deployment blocked because quality gate failed.")

    for name in ["model.joblib", "scaler.joblib", "metrics.json"]:
        shutil.copy2(MODEL_DIR / name, PROD_DIR / name)

    metadata = {
        "status": "DEPLOYED",
        "deployment_timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "v1.0.0",
        "deployed_files": [
            "model.joblib",
            "scaler.joblib",
            "metrics.json",
            "training_metadata.json"
        ],
        "metrics": {
            "accuracy": metrics["accuracy"],
            "passed_quality_gate": metrics["passed_quality_gate"]
        }
    }
    (PROD_DIR / "training_metadata.json").write_text(json.dumps(
        {"model_type": "RandomForestClassifier", "random_state": 42}, indent=2
    ))
    (PROD_DIR / "model_manifest.json").write_text(json.dumps(metadata, indent=2))
    print("Model successfully deployed to 'production_model'")

if __name__ == "__main__":
    deploy_model()
