PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name TEXT NOT NULL,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('admin', 'teacher')),
    department TEXT,
    subject TEXT,
    approval_status TEXT NOT NULL DEFAULT 'pending'
        CHECK(approval_status IN ('pending', 'approved', 'rejected', 'inactive')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
CREATE INDEX IF NOT EXISTS idx_users_status ON users(approval_status);

CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT UNIQUE NOT NULL,
    student_name TEXT NOT NULL,
    department TEXT NOT NULL,
    semester INTEGER NOT NULL,
    academic_year TEXT NOT NULL,
    assigned_teacher_id INTEGER,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (assigned_teacher_id) REFERENCES users(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_students_teacher ON students(assigned_teacher_id);
CREATE INDEX IF NOT EXISTS idx_students_dept ON students(department);

CREATE TABLE IF NOT EXISTS academic_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    previous_semester_marks REAL,
    previous_sgpa REAL,
    previous_backlogs INTEGER DEFAULT 0,
    ia1_score REAL, ia1_max_marks REAL DEFAULT 30,
    ia2_score REAL, ia2_max_marks REAL DEFAULT 30,
    ia3_score REAL, ia3_max_marks REAL DEFAULT 30,
    attendance REAL,
    assignment_average REAL,
    assignment_completion REAL,
    quiz_average REAL,
    study_hours REAL,
    actual_final_marks REAL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_records_student ON academic_records(student_id);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    teacher_id INTEGER NOT NULL,
    predicted_marks REAL NOT NULL,
    performance_category TEXT NOT NULL,
    risk_probability REAL,
    selected_model TEXT NOT NULL,
    prediction_date TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE,
    FOREIGN KEY (teacher_id) REFERENCES users(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_predictions_student ON predictions(student_id);
CREATE INDEX IF NOT EXISTS idx_predictions_teacher ON predictions(teacher_id);

CREATE TABLE IF NOT EXISTS prediction_explanations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL,
    feature_name TEXT NOT NULL,
    feature_value REAL,
    shap_contribution REAL,
    explanation TEXT,
    FOREIGN KEY (prediction_id) REFERENCES predictions(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_explanations_pred ON prediction_explanations(prediction_id);

CREATE TABLE IF NOT EXISTS model_registry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    model_type TEXT NOT NULL CHECK(model_type IN ('ml', 'dl')),
    evaluation_metrics TEXT,
    training_date TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    active_status INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_registry_active ON model_registry(active_status);

CREATE TABLE IF NOT EXISTS activity_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    action TEXT NOT NULL,
    details TEXT,
    timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_logs_user ON activity_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_logs_time ON activity_logs(timestamp);