"""Test database integrity."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.database import query


def test_all_tables_exist():
    expected = {"users", "students", "academic_records", "predictions",
                "prediction_explanations", "model_registry", "activity_logs"}
    rows = query("SELECT name FROM sqlite_master WHERE type='table'")
    actual = {r["name"] for r in rows} - {"sqlite_sequence"}
    missing = expected - actual
    assert not missing, f"Missing tables: {missing}"


def test_foreign_keys_enforced():
    rows = query("PRAGMA foreign_keys")
    assert rows[0]["foreign_keys"] == 1, "Foreign keys must be ON"


if __name__ == "__main__":
    test_all_tables_exist()
    test_foreign_keys_enforced()
    print("[PASS] Database tests passed")