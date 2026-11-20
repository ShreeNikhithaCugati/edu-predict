"""Recommendation engine — rule-based, informed by academic data + SHAP."""
from collections import Counter
from typing import List

# Same weights the dynamic model was trained with (see retrain_final.py).
IA_WEIGHT_PCT = {"IA1": 40, "IA2": 20, "IA3": 10}
IA_MAX = 50.0


def _subject_recommendations(subject_impacts: List[dict]) -> List[str]:
    """Recommendations built from the subject-wise IA bar graph."""
    if not subject_impacts:
        return []

    rows = {row["Subject"]: row for row in subject_impacts}
    scores = {subject: float(row.get("Weighted %", 0)) for subject, row in rows.items()}
    focuses = {subject: row.get("Focus", "IA1") for subject, row in rows.items()}
    total = len(scores)
    if total == 0:
        return []

    recs = []

    # Which test is the shared weak spot?
    focus_counts = Counter(focuses.values())
    focus, focus_count = focus_counts.most_common(1)[0]
    weight = IA_WEIGHT_PCT.get(focus, 0)
    gain_per_subject = weight / 100.0 * (5.0 / IA_MAX * 100.0)

    if focus_count == total:
        recs.append(
            f"📚 **{focus} is the weakest component in all {total} subjects** — "
            f"it carries {weight}% of the final marks. Raising {focus} by 5 marks "
            f"in every subject would add about **+{gain_per_subject:.1f} marks**."
        )
    else:
        affected = [subject for subject in scores if focuses[subject] == focus]
        if len(affected) <= 4:
            listed = ", ".join(affected)
        else:
            listed = ", ".join(affected[:3]) + f" and {len(affected) - 3} more"
        gain = gain_per_subject * focus_count / total
        recs.append(
            f"📚 **{focus} is the weakest component in {focus_count} of {total} subjects** "
            f"({listed}). Improving it by 5 marks there would add about **+{gain:.1f} marks**."
        )

    # Strongest / weakest subject
    weakest = min(scores, key=scores.get)
    strongest = max(scores, key=scores.get)

    if scores[strongest] - scores[weakest] < 1.0:
        recs.append(
            f"⚖️ **All {total} subjects are performing equally "
            f"({scores[weakest]:.0f}% weighted)** — there is no single weak subject, "
            "so spread revision time evenly and target the shared weak test above."
        )
    else:
        weak_row = rows[weakest]
        weak_mark = float(weak_row.get(focuses[weakest], 0))
        recs.append(
            f"🎯 **{weakest} is the weakest subject ({scores[weakest]:.0f}% weighted)** — "
            f"start with {focuses[weakest]} ({weak_mark:.0f}/{IA_MAX:.0f})."
        )
        if scores[strongest] >= 75:
            recs.append(
                f"✅ **{strongest} is your strongest subject ({scores[strongest]:.0f}%)** — "
                "keep the same routine there and reuse it for the weaker subjects."
            )

    return recs


def generate_recommendations(
    record: dict,
    shap_contributions: List[dict] | None = None,
    subject_impacts: List[dict] | None = None,
) -> List[str]:
    """
    record: dict of academic_records row (ia1_score, attendance, etc.)
    shap_contributions: list of {feature, value, shap} from Phase 5 prediction
    subject_impacts: list of subject rows from the IA bar graph
                     (Subject, IA1..IA3, Weighted %, Focus, Status)
    Returns: list of recommendation strings.
    """
    recs = []

    # --- attendance ---
    att = record.get("attendance") or 0
    if att < 60:
        recs.append(
            f"⚠️ **Attendance is critically low ({att:.0f}%)** — below 60%. "
            "Immediate improvement is essential; most institutions require 75% minimum."
        )
    elif att < 75:
        recs.append(
            f"📉 **Attendance below 75% ({att:.0f}%)** — prioritize attending all remaining classes."
        )

    # --- internal assessment trend ---
    ia1 = record.get("ia1_score") or 0
    ia2 = record.get("ia2_score") or 0
    ia3 = record.get("ia3_score") or 0
    ia_max = record.get("ia1_max_marks") or 30

    if ia_max > 0:
        ia_pct = [ia1 / ia_max * 100, ia2 / ia_max * 100, ia3 / ia_max * 100]
        ia_avg = sum(ia_pct) / 3

        if ia_avg < 50:
            recs.append(
                f"📚 **IA average is {ia_avg:.0f}%** — invest in subject fundamentals. "
                "Recommended: extra tutorials, peer study groups, and focused revision."
            )

        # declining trend
        if ia3 < ia1 - 5:
            recs.append(
                f"📉 **Declining IA trend** — IA3 ({ia3:.0f}) dropped by {ia1 - ia3:.0f} marks vs IA1 ({ia1:.0f}). "
                "Early intervention recommended: talk to the student about workload or difficulties."
            )
        elif ia3 > ia1 + 5:
            recs.append(
                f"📈 **Improving IA trend** — IA3 ({ia3:.0f}) improved by {ia3 - ia1:.0f} marks vs IA1 ({ia1:.0f}). "
                "Positive progress; encourage continued effort."
            )

    # --- assignment ---
    asg_avg = record.get("assignment_average") or 0
    asg_compl = record.get("assignment_completion") or 0
    if asg_avg < 60:
        recs.append(
            f"📝 **Assignment average is {asg_avg:.0f}%** — strengthen practical work. "
            "Set weekly assignment targets and seek feedback."
        )
    if asg_compl < 70:
        recs.append(
            f"📋 **Assignment completion at {asg_compl:.0f}%** — incomplete submissions hurt final grades. "
            "Create a submission calendar."
        )

    # --- quiz ---
    quiz = record.get("quiz_average") or 0
    if quiz < 60:
        recs.append(
            f"🧪 **Quiz average is {quiz:.0f}%** — practice regular short-form assessments. "
            "Weekly self-quizzes on recent topics will help."
        )

    # --- previous performance ---
    prev = record.get("previous_semester_marks") or 0
    if prev < 55:
        recs.append(
            f"📖 **Previous semester marks were {prev:.0f}%** — foundational gaps may be affecting current performance. "
            "Consider remedial sessions on prerequisite topics."
        )

    # --- backlogs ---
    bl = record.get("previous_backlogs") or 0
    if bl >= 2:
        recs.append(
            f"🎯 **{bl} previous backlog(s)** — clearing these is critical. "
            "Prioritize backlog exams alongside current coursework."
        )

    # --- study hours ---
    study = record.get("study_hours") or 0
    if study < 2:
        recs.append(
            f"⏰ **Study hours are low ({study:.1f} hrs/day)** — target at least 3–4 hours of focused study daily."
        )

    # --- subject-wise recommendations (from the IA bar graph) ---
    recs.extend(_subject_recommendations(subject_impacts or []))

    # --- SHAP-informed recommendation (behavioural factors) ---
    if shap_contributions:
        negatives = sorted(
            [s for s in shap_contributions if float(s.get("SHAP Value", 0)) < -0.5],
            key=lambda s: float(s["SHAP Value"]),
        )
        seen = set()
        for contribution in negatives:
            readable = contribution.get("Display Feature") or contribution.get(
                "Feature", ""
            ).replace("_", " ").title()
            if readable in seen:
                continue
            seen.add(readable)
            recs.append(
                f"🔍 **{readable} is pulling the prediction down** "
                f"(contribution: {float(contribution['SHAP Value']):+.1f} marks). "
                "Focused improvement here will have the highest impact."
            )
            if len(seen) >= 2:
                break

    if not recs:
        recs.append("✅ **No specific weak areas detected.** Continue current study habits and maintain consistency.")

    return recs
