"""
Reset a user's password (and approve the account).

Usage:
    python reset_password.py --list
    python reset_password.py <username> <new_password>

Examples:
    python reset_password.py --list
    python reset_password.py admin MyNewPass123
    python reset_password.py T1 Teacher12345
"""
import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

from authentication.password_security import hash_password
from authentication.auth_service import find_user_by_username
from database.database import query, execute


def list_users():
    rows = query("SELECT id, username, role, approval_status FROM users ORDER BY id")
    if not rows:
        print("No users found. Run: python -m database.init_db")
        return
    print(f"{'ID':>3}  {'USERNAME':<22} {'ROLE':<8} STATUS")
    print("-" * 55)
    for r in rows:
        print(f"{r['id']:>3}  {r['username']:<22} {r['role']:<8} {r['approval_status']}")


def reset(username: str, new_password: str):
    user = find_user_by_username(username)
    if not user:
        print(f"[ERROR] No user named '{username}'. Use --list to see users.")
        sys.exit(1)
    execute(
        "UPDATE users SET password_hash = ?, approval_status = 'approved' WHERE username = ?",
        (hash_password(new_password), username),
    )
    print(f"[OK] Password for '{username}' reset and account set to 'approved'.")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
    elif args[0] == "--list":
        list_users()
    elif len(args) == 2:
        reset(args[0], args[1])
    else:
        print("Usage:\n  python reset_password.py --list\n  python reset_password.py <username> <new_password>")
