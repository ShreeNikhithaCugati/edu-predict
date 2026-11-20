"""Pre-deployment check: verifies every file the web app needs at runtime.

Run from the project root:

    python deploy_check.py

Anything listed as MISSING would make the deployed app fail, because
Streamlit Community Cloud only has the files that are committed to Git.
"""
import os

REQUIRED_FILES = [
    "app.py",
    "requirements.txt",
    "database/student_perf.db",
    "database/schema.sql",
    "saved_models/dynamic_xgb_model.pkl",
    "saved_models/dynamic_features.json",
    "saved_models/xgboost_model.pkl",
    "saved_models/randomforest_model.pkl",
    "saved_models/scaler.pkl",
    "data/raw/training_data.csv",
]

REQUIRED_DIRS = [
    "data/subject_records",
]


def main() -> int:
    missing = []

    print("Files:")
    for path in REQUIRED_FILES:
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"  OK       {path}  ({size / 1024:.1f} KB)")
        else:
            print(f"  MISSING  {path}")
            missing.append(path)

    print("\nFolders:")
    for path in REQUIRED_DIRS:
        if os.path.isdir(path) and os.listdir(path):
            print(f"  OK       {path}/  ({len(os.listdir(path))} files)")
        else:
            print(f"  MISSING  {path}/")
            missing.append(path)

    print()
    if missing:
        print(f"{len(missing)} item(s) missing — commit them before deploying.")
        return 1

    print("All runtime files present. Ready to deploy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
