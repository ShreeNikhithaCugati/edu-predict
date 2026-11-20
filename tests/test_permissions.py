"""Test that roles and statuses are enforced."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.database import query


def test_admin_exists():
    rows = query("SELECT id FROM users WHERE role='admin' AND approval_status='approved'")
    assert len(rows) >= 1, "At least one approved admin must exist"


def test_roles_are_valid():
    rows = query("SELECT DISTINCT role FROM users")
    roles = {r["role"] for r in rows}
    assert roles.issubset({"admin", "teacher"}), f"Unexpected roles: {roles}"


def test_statuses_are_valid():
    rows = query("SELECT DISTINCT approval_status FROM users")
    statuses = {r["approval_status"] for r in rows}
    allowed = {"pending", "approved", "rejected", "inactive"}
    assert statuses.issubset(allowed), f"Unexpected statuses: {statuses}"


if __name__ == "__main__":
    test_admin_exists()
    test_roles_are_valid()
    test_statuses_are_valid()
    print("[PASS] Permission tests passed")