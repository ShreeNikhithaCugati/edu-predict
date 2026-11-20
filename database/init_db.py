"""Creates tables and first admin. Run: python -m database.init_db"""
import os
import sys
from dotenv import load_dotenv
from argon2 import PasswordHasher

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.database import get_db, query, execute

load_dotenv()
ph = PasswordHasher()


def init_schema():
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = f.read()
    with get_db() as conn:
        conn.executescript(schema)
    print("[OK] schema created")


def create_admin_if_missing():
    username = os.getenv("ADMIN_USERNAME", "admin")
    email = os.getenv("ADMIN_EMAIL", "admin@college.edu")
    password = os.getenv("ADMIN_PASSWORD")

    if not password:
        print("[SKIP] ADMIN_PASSWORD not set in .env")
        return

    existing = query("SELECT id FROM users WHERE username = ?", (username,))
    if existing:
        print(f"[SKIP] admin '{username}' already exists")
        return

    password_hash = ph.hash(password)
    execute(
        """INSERT INTO users
           (full_name, username, email, password_hash, role, approval_status)
           VALUES (?, ?, ?, ?, 'admin', 'approved')""",
        ("System Administrator", username, email, password_hash),
    )
    print(f"[OK] admin '{username}' created")


if __name__ == "__main__":
    init_schema()
    create_admin_if_missing()
    print("\nDatabase ready at", os.getenv("DB_PATH", "database/student_perf.db"))