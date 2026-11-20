"""
Generates synthetic student test data.
Long format: one row = one (student, test_no).
"""
import numpy as np
import pandas as pd

def generate_dataset(n_students=800, n_tests=4, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for sid in range(n_students):
        ability = rng.normal(60, 15)
        trend = rng.normal(0, 3)
        att_base = np.clip(rng.normal(0.85, 0.10), 0.3, 1.0)
        asg_base = np.clip(rng.normal(70, 12), 0, 100)
        part_base = np.clip(rng.normal(3, 1), 1, 5)

        for t in range(1, n_tests + 1):
            marks = np.clip(ability + trend * (t - 1) + rng.normal(0, 6), 0, 100)
            att = np.clip(att_base + rng.normal(0, 0.05), 0, 1)
            asg = np.clip(asg_base + rng.normal(0, 6), 0, 100)
            part = np.clip(part_base + rng.normal(0, 0.5), 1, 5)
            rows.append({
                "student_id": f"S{sid:04d}",
                "test_no": t,
                "marks": round(marks, 2),
                "attendance": round(att, 3),
                "assignment": round(asg, 2),
                "participation": round(part, 2),
            })

    df = pd.DataFrame(rows)

    agg = df.groupby("student_id").agg(
        last_marks=("marks", "last"),
        mean_marks=("marks", "mean"),
        mean_att=("attendance", "mean"),
        mean_asg=("assignment", "mean"),
        mean_part=("participation", "mean"),
    ).reset_index()

    final = (0.45 * agg["last_marks"]
             + 0.30 * agg["mean_marks"]
             + 15 * agg["mean_att"]
             + 0.20 * agg["mean_asg"]
             + 2 * agg["mean_part"]
             + rng.normal(0, 4, len(agg)))
    agg["final_score"] = np.clip(final, 0, 100).round(2)

    return df, agg[["student_id", "final_score"]]


if __name__ == "__main__":
    df, labels = generate_dataset()
    df.to_csv("data/tests.csv", index=False)
    labels.to_csv("data/labels.csv", index=False)
    print(f"Wrote data/tests.csv ({len(df)} rows) and data/labels.csv ({len(labels)} rows)")