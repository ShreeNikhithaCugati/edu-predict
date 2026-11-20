"""Authentication business logic — DB queries + password checks."""
from database.database import query, execute
from authentication.password_security import hash_password, verify_password, is_strong_password


def find_user_by_username(username: str):
    rows = query("SELECT * FROM users WHERE username = ?", (username,))
    return rows[0] if rows else None


def find_user_by_email(email: str):
    rows = query("SELECT * FROM users WHERE email = ?", (email,))
    return rows[0] if rows else None


def attempt_login(username: str, password: str) -> tuple[bool, str, dict | None]:
    """Returns (success, message, user_dict)."""
    user = find_user_by_username(username)
    if not user:
        return False, "Invalid username or password.", None

    if user["approval_status"] == "pending":
        return False, "Your account is awaiting admin approval.", None
    if user["approval_status"] == "rejected":
        return False, "Your account was rejected. Contact the administrator.", None
    if user["approval_status"] == "inactive":
        return False, "Your account is inactive.", None

    # Verify the password against the stored hash
    if not verify_password(password, user["password_hash"]):
        return False, "Invalid username or password.", None

    # strip password hash before returning
    safe_user = {k: v for k, v in user.items() if k != "password_hash"}
    log_activity(user["id"], "login", f"{username} logged in")
    return True, "Login successful.", safe_user


def register_teacher(full_name: str, teacher_id: str, email: str,
                     username: str, department: str, subject: str,
                     password: str, confirm: str) -> tuple[bool, str]:
    # validation
    if not all([full_name, teacher_id, email, username, department, subject, password]):
        return False, "All fields are required."

    if password != confirm:
        return False, "Passwords do not match."

    ok, msg = is_strong_password(password)
    if not ok:
        return False, msg

    if find_user_by_username(username):
        return False, "Username already taken."
    if find_user_by_email(email):
        return False, "Email already registered."

    # HASH THE PASSWORD BEFORE SAVING
    pwd_hash = hash_password(password)
    
    # Save the user as 'pending'
    execute(
        """INSERT INTO users
           (full_name, username, email, password_hash, role,
            department, subject, approval_status)
           VALUES (?, ?, ?, ?, 'teacher', ?, ?, 'pending')""",
        (full_name, username, email, pwd_hash, department, subject),
    )
    return True, "Registration submitted. Await admin approval."


def log_activity(user_id: int | None, action: str, details: str = ""):
    execute(
        "INSERT INTO activity_logs (user_id, action, details) VALUES (?, ?, ?)",
        (user_id, action, details),
    )