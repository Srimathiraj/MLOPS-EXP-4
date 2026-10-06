import subprocess
import sys

stages = [
    ("STAGE 1: Running Unit Tests (CI)", [sys.executable, "-m", "pytest", "tests/", "-v"]),
    ("STAGE 2: Preprocessing Data", [sys.executable, "src/data_preprocessing.py"]),
    ("STAGE 3: Training Model", [sys.executable, "src/train.py"]),
    ("STAGE 4: Evaluating Model & Validating Quality Gate", [sys.executable, "src/evaluate.py"]),
    ("STAGE 5: Deploying Validated Model", [sys.executable, "src/deploy.py"]),
    ("STAGE 6: Running Production Inference Smoke Test", [sys.executable, "src/predict.py"]),
]

for title, command in stages:
    print(f"\n>>> {title}...")
    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"FAILED: {title}")
        sys.exit(result.returncode)

print("\nSUCCESS: All pipeline stages completed successfully!")
