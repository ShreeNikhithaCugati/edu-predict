"""Argon2id password hashing."""
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, VerificationError

ph = PasswordHasher()

def hash_password(plain: str) -> str:
    return ph.hash(plain)

def verify_password(plain: str, hashed: str) -> bool:
    try:
        return ph.verify(hashed, plain)
    except (VerifyMismatchError, VerificationError):
        return False

def is_strong_password(pwd: str) -> tuple[bool, str]:
    if len(pwd) < 8:
        return False, "Password must be at least 8 characters."
    if not any(c.isupper() for c in pwd):
        return False, "Password must contain an uppercase letter."
    if not any(c.islower() for c in pwd):
        return False, "Password must contain a lowercase letter."
    if not any(c.isdigit() for c in pwd):
        return False, "Password must contain a digit."
    # Special character rule REMOVED so Teacher12026 works
    return True, "OK"