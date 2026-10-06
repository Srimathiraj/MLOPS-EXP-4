from src.data_preprocessing import preprocess_data
from pathlib import Path

def test_preprocess_data():
    preprocess_data()
    assert (Path("data/processed/dataset.joblib")).exists()
