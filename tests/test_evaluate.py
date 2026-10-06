from src.data_preprocessing import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model
import json

def test_evaluate_model():
    preprocess_data()
    train_model()
    evaluate_model()
    metrics = json.load(open("models/metrics.json"))
    assert metrics["accuracy"] >= 0.85
    assert metrics["passed_quality_gate"] is True
