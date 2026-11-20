"""Generate synthetic training data matching the new schema. Run once."""
import os
import numpy as np
import pandas as pd

FEATURE_COLS = [
    "previous_semester_marks", "previous_sgpa", "previous_backlogs",
    "ia1_score", "ia2_score", "ia3_score",
    "attendance", "assignment_average", "assignment_completion",
    "quiz_average", "study_hours",
]
TARGET = "actual_final_marks"
OUT = "data/raw/training_data.csv"


def generate(n=1000, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        prev_marks = float(np.clip(rng.normal(65, 15), 20, 100))
        prev_sgpa = float(np.clip(rng.normal(6.5, 1.5), 2, 10))
        backlogs = int(rng.poisson(0.5))

        base = prev_marks + rng.normal(0, 5)
        ia1 = float(np.clip(base + rng.normal(0, 6), 0, 30))
        ia2 = float(np.clip(base + rng.normal(0, 6), 0, 30))
        ia3 = float(np.clip(base + rng.normal(0, 6), 0, 30))

        attendance = float(np.clip(rng.normal(80, 12), 30, 100))
        assignment_avg = float(np.clip(rng.normal(70, 15), 0, 100))
        assignment_compl = float(np.clip(rng.normal(75, 15), 0, 100))
        quiz_avg = float(np.clip(rng.normal(65, 15), 0, 100))
        study_hours = float(np.clip(rng.normal(3.5, 1.8), 0.5, 12))

        # synthetic target: combination weighted by intuition
        ia_avg = (ia1 + ia2 + ia3) / 3
        final = (
            0.30 * prev_marks
            + 0.20 * (prev_sgpa * 10)
            + 0.30 * (ia_avg / 30 * 100)
            + 0.10 * attendance
            + 0.05 * assignment_avg
            + 0.05 * (quiz_avg)
            - 2 * backlogs
            + 0.5 * study_hours
            + rng.normal(0, 4)
        )
        final = float(np.clip(final, 0, 100))

        rows.append({
            "student_id": f"S{i:04d}",
            "previous_semester_marks": round(prev_marks, 2),
            "previous_sgpa": round(prev_sgpa, 2),
            "previous_backlogs": backlogs,
            "ia1_score": round(ia1, 2),
            "ia2_score": round(ia2, 2),
            "ia3_score": round(ia3, 2),
            "attendance": round(attendance, 2),
            "assignment_average": round(assignment_avg, 2),
            "assignment_completion": round(assignment_compl, 2),
            "quiz_average": round(quiz_avg, 2),
            "study_hours": round(study_hours, 2),
            "actual_final_marks": round(final, 2),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    df = generate()
    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT} — {len(df)} rows × {len(df.columns)} cols")