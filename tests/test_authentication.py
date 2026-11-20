"""Test password hashing and auth logic."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from authentication.password_security import hash_password, verify_password, is_strong_password


def test_password_hashing():
    pwd = "TestPass123!"
    h = hash_password(pwd)
    assert h != pwd, "Password must not be stored in plaintext"
    assert h.startswith("$argon2id$"), "Must use Argon2id"
    assert verify_password(pwd, h), "Correct password must verify"
    assert not verify_password("WrongPass", h), "Wrong password must fail"


def test_password_strength():
    ok, _ = is_strong_password("short")
    assert not ok
    ok, _ = is_strong_password("alllowercase1")
    assert not ok
    ok, _ = is_strong_password("ALLUPPERCASE1")
    assert not ok
    ok, _ = is_strong_password("NoDigitsHere")
    assert not ok
    ok, _ = is_strong_password("GoodPass1")
    assert ok


if __name__ == "__main__":
    test_password_hashing()
    test_password_strength()
    print("[PASS] Authentication tests passed")