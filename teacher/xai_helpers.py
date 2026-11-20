"""Subject-aware helpers for the teacher prediction page.

The dynamic XGBoost model was trained by copying one IA score into every
subject column (see retrain_final.py).  As a result the model learns an
"IA level", not subject-specific effects: every subject column is a
duplicate of the same three numbers.

These helpers keep the UI honest and aligned with the IA bar graph:

* rebuild the model row exactly the way training data looked,
* group the model's subject SHAP back onto the student's real subjects,
* build the subject table shown under the bar chart,
* run LIME on a compact subject view so its labels are real subject names.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

IA_MAX = 50.0
IA_WEIGHTS = {"IA1": 0.40, "IA2": 0.20, "IA3": 0.10}

BEHAVIOUR_LABELS = {
    "previous_semester_marks": "Previous semester marks",
    "previous_sgpa": "Previous SGPA",
    "previous_backlogs": "Previous backlogs",
    "attendance": "Attendance",
    "assignment_average": "Assignment average",
    "quiz_average": "Quiz average",
    "study_hours": "Study hours / day",
}


def is_subject_feature(name: str) -> bool:
    """True for columns such as ``Python Programming_IA2``."""
    return name.endswith(("_IA1", "_IA2", "_IA3"))


def weighted_score(marks) -> float:
    """0-100 score using the same 40/20/10 weights the model was trained on."""
    weighted = sum(
        IA_WEIGHTS[f"IA{i + 1}"] * (float(marks[i]) / IA_MAX * 100.0)
        for i in range(3)
    )
    return weighted / sum(IA_WEIGHTS.values())


def per_test_average(subject_marks: dict) -> list[float]:
    if not subject_marks:
        return [0.0, 0.0, 0.0]
    return [
        float(np.mean([m[i] for m in subject_marks.values() if len(m) > i]))
        for i in range(3)
    ]


def build_model_input(base_values: dict, subject_marks: dict, features: list[str]):
    """Build one model row the way the training data looked.

    Training copied the same IA1/IA2/IA3 into every subject column, so the
    faithful input is the student's per-test average in every subject column.
    Filling the other columns with zero (the old behaviour) made the model
    believe the student had skipped 48 subjects and produced a bogus
    "Applied Physics" explanation.
    """
    avg = per_test_average(subject_marks)
    row = {k: float(v) if v is not None else 0.0 for k, v in base_values.items()}
    for feature in features:
        if feature in row:
            continue
        if is_subject_feature(feature):
            row[feature] = avg[int(feature.rsplit("_IA", 1)[1]) - 1]
        else:
            row[feature] = 0.0
    return pd.DataFrame([row])[features].to_numpy(), avg


def subject_table(subject_marks: dict) -> pd.DataFrame:
    """One row per subject, ready to display under the IA bar chart."""
    rows = []
    for subject, marks in subject_marks.items():
        marks = [float(m) for m in marks]
        score = weighted_score(marks)
        weakest = f"IA{int(np.argmin(marks)) + 1}"
        rows.append(
            {
                "Subject": subject,
                "IA1": marks[0],
                "IA2": marks[1],
                "IA3": marks[2],
                "Weighted %": round(score, 1),
                "Status": "Strong" if score >= 75 else ("On track" if score >= 60 else "Needs work"),
                "Focus": weakest,
            }
        )
    return pd.DataFrame(rows)


def split_shap(features, values, shap_values, subject_marks: dict):
    """Split SHAP into behavioural rows and the model's total subject effect."""
    behaviour = []
    subject_total = 0.0
    for feature, value, shap_value in zip(features, values, shap_values):
        if is_subject_feature(feature):
            subject_total += float(shap_value)
        else:
            behaviour.append(
                {
                    "Feature": feature,
                    "Display Feature": BEHAVIOUR_LABELS.get(
                        feature, feature.replace("_", " ").title()
                    ),
                    "Value": float(value),
                    "SHAP Value": float(shap_value),
                }
            )
    return behaviour, subject_total


def subject_shap_rows(subject_marks: dict, subject_total: float) -> list[dict]:
    """Spread the model's subject block over the student's real subjects.

    The model cannot distinguish subjects (their columns were identical during
    training), so the *total* subject effect is exact while the split is a
    reasonable allocation: weaker subjects carry more of a negative effect and
    stronger subjects get more of a positive one.
    """
    scores = {subject: weighted_score(marks) for subject, marks in subject_marks.items()}
    if not scores:
        return []

    if subject_total >= 0:
        raw = {subject: max(score, 0.0) for subject, score in scores.items()}
    else:
        best = max(scores.values())
        raw = {subject: max(best - score, 0.0) for subject, score in scores.items()}

    total_raw = sum(raw.values())
    if total_raw <= 1e-9:  # all subjects equal - split evenly
        raw = {subject: 1.0 for subject in scores}
        total_raw = float(len(scores))

    return [
        {
            "Feature": f"{subject}_subject",
            "Display Feature": subject,
            "Value": scores[subject],
            "SHAP Value": subject_total * raw[subject] / total_raw,
        }
        for subject in scores
    ]


def shap_frame(behaviour: list[dict], subject_rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(behaviour + subject_rows)
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["Impact"] = frame["SHAP Value"].apply(
        lambda value: "Positive (Helps)" if value > 0 else "Negative (Hurts)"
    )
    frame = frame.reindex(frame["SHAP Value"].abs().sort_values().index)
    return frame


def grouped_lime_inputs(base_values: dict, subject_marks: dict, features: list[str]):
    """Build the compact LIME view: 7 behavioural columns + 6x3 subject marks.

    LIME is run on this compact view instead of the model's 169 duplicated
    columns, so every explanation is labelled with the student's real subjects.
    """
    behaviour_features = [f for f in features if not is_subject_feature(f)]
    subject_names = list(subject_marks)

    behaviour_values = np.array(
        [float(base_values.get(f, 0.0) or 0.0) for f in behaviour_features]
    )
    subject_values = np.array(
        [[float(value) for value in subject_marks[subject]] for subject in subject_names]
    )

    instance = np.concatenate([behaviour_values, subject_values.ravel()])
    names = [
        BEHAVIOUR_LABELS.get(feature, feature.replace("_", " ").title())
        for feature in behaviour_features
    ]
    names += [
        f"{subject} — {test} ({float(subject_marks[subject][int(test[-1]) - 1]):.0f}/{IA_MAX:.0f})"
        for subject in subject_names
        for test in ("IA1", "IA2", "IA3")
    ]

    return instance, names, behaviour_features, subject_names


def make_grouped_predictor(model, features: list[str], n_subjects: int):
    """Wrap the model so it accepts the compact LIME view.

    The model only ever learned one common IA level, so every subject column is
    set to the mean of the six subject values. That keeps the wrapper's
    prediction identical to the real model while exposing subject features.
    """
    behaviour_idx = [i for i, feature in enumerate(features) if not is_subject_feature(feature)]
    subject_idx = {
        test: [i for i, feature in enumerate(features) if feature.endswith(f"_{test}")]
        for test in ("IA1", "IA2", "IA3")
    }
    n_behaviour = len(behaviour_idx)
    template = np.zeros((1, len(features)))

    def predict(matrix):
        matrix = np.atleast_2d(np.asarray(matrix, dtype=float))
        rows = len(matrix)
        full = np.tile(template, (rows, 1))
        full[:, behaviour_idx] = matrix[:, :n_behaviour]

        subject_block = matrix[:, n_behaviour:].reshape(rows, n_subjects, 3)
        mean_ia = subject_block.mean(axis=1)
        for column, test in enumerate(("IA1", "IA2", "IA3")):
            full[:, subject_idx[test]] = mean_ia[:, column][:, None]

        return model.predict(full)

    return predict


def grouped_lime_background(
    training_frame: pd.DataFrame,
    behaviour_features: list[str],
    n_subjects: int,
    sample_size: int = 150,
):
    """Realistic LIME background in the compact view."""
    sample = training_frame.sample(
        min(sample_size, len(training_frame)), random_state=42
    ).reset_index(drop=True)

    behaviour = sample[behaviour_features].to_numpy(dtype=float)
    ia_scores = np.column_stack(
        [
            sample["ia1_score"] / 30.0 * IA_MAX,
            sample["ia2_score"] / 30.0 * IA_MAX,
            sample["ia3_score"] / 30.0 * IA_MAX,
        ]
    )
    subject_block = np.repeat(ia_scores[:, None, :], n_subjects, axis=1).reshape(
        len(sample), n_subjects * 3
    )

    # Training IA scores are almost all 50/50, which collapses LIME's
    # discretisation into a single bin (every subject weight becomes 0).
    # A small jitter keeps the cohort mean but gives LIME usable bands.
    rng = np.random.default_rng(42)
    subject_block = np.clip(
        subject_block + rng.normal(0, 8.0, size=subject_block.shape), 0.0, IA_MAX
    )

    return np.hstack([behaviour, subject_block])


def lime_rows_from_explanation(explanation, names: list[str], limit: int = 6) -> list[dict]:
    """Read LIME's feature indices directly (no fragile string parsing)."""
    mapping = explanation.as_map()
    if isinstance(mapping, dict):
        # Regression explanations are stored under dummy_label (1), not 0.
        label = getattr(explanation, "dummy_label", 1)
        pairs = mapping.get(label, mapping.get(0, []))
    else:
        pairs = mapping

    rows = []
    for index, weight in pairs:
        weight = float(weight)
        if abs(weight) <= 0.001:
            continue
        rows.append({"label": names[int(index)], "weight": weight})

    rows.sort(key=lambda row: abs(row["weight"]), reverse=True)
    return rows[:limit]