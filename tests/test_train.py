from src.data_preprocessing import preprocess_data
from src.train import train_model
from pathlib import Path

def test_train_model():
    preprocess_data()
    train_model()
    assert Path("models/model.joblib").exists()
    assert Path("models/scaler.joblib").exists()
