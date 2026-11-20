"""Plotly chart helpers."""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib
import os
import json
from database.database import query


def performance_distribution_chart(dept=None, sem=None):
    """Pie chart of performance categories across all predictions."""
    sql = """
        SELECT p.performance_category, COUNT(*) AS count
        FROM predictions p
        JOIN students s ON s.student_id = p.student_id
        WHERE 1=1
    """
    params = []
    if dept and dept != "All":
        sql += " AND s.department = ?"
        params.append(dept)
    if sem and sem != "All":
        sql += " AND s.semester = ?"
        params.append(sem)
    sql += " GROUP BY p.performance_category"

    rows = query(sql, tuple(params))
    if not rows:
        return None
    df = pd.DataFrame(rows)
    fig = px.pie(df, values="count", names="performance_category",
                 title="Performance Distribution",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    return fig


def department_performance_chart(dept=None, sem=None):
    """Average predicted marks per department (Only shows departments with predictions)."""
    sql = """
        SELECT UPPER(TRIM(s.department)) AS department,
               AVG(p.predicted_marks) AS avg_marks,
               COUNT(p.id) AS n
        FROM predictions p
        JOIN students s ON s.student_id = p.student_id
        WHERE 1=1
    """
    params = []
    if dept and dept != "All":
        sql += " AND UPPER(TRIM(s.department)) = UPPER(TRIM(?))"
        params.append(dept)
    if sem and sem != "All":
        sql += " AND s.semester = ?"
        params.append(sem)
    sql += " GROUP BY UPPER(TRIM(s.department)) ORDER BY avg_marks DESC"

    rows = query(sql, tuple(params))
    if not rows:
        return None

    df = pd.DataFrame(rows)
    if df.empty:
        return None

    fig = px.bar(
        df,
        x="department",
        y="avg_marks",
        title="Average Predicted Marks by Department",
        text="avg_marks"
    )
    fig.update_traces(
        marker_color="#3498db",
        texttemplate="%{text:.1f}",
        textposition="outside"
    )
    fig.update_layout(yaxis_range=[0, 100], showlegend=False)
    return fig


def risk_distribution_chart(dept=None, sem=None):
    """Bar chart of at-risk vs on-track students."""
    sql = """
        SELECT p.performance_category, COUNT(*) AS count
        FROM predictions p
        JOIN students s ON s.student_id = p.student_id
        WHERE 1=1
    """
    params = []
    if dept and dept != "All":
        sql += " AND s.department = ?"
        params.append(dept)
    if sem and sem != "All":
        sql += " AND s.semester = ?"
        params.append(sem)
    sql += " GROUP BY p.performance_category"

    rows = query(sql, tuple(params))
    if not rows:
        return None
    df = pd.DataFrame(rows)
    color_map = {"Excellent": "#2ecc71", "Good": "#3498db",
                 "Average": "#f39c12", "At Risk": "#e74c3c"}
    fig = px.bar(df, x="performance_category", y="count",
                 title="Risk Distribution",
                 color="performance_category",
                 color_discrete_map=color_map)
    return fig


def predictions_over_time_chart():
    """Line chart of predictions per day."""
    rows = query("""
        SELECT SUBSTR(prediction_date, 1, 10) AS day, COUNT(*) AS count
        FROM predictions GROUP BY SUBSTR(prediction_date, 1, 10)
        ORDER BY day
    """)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    fig = px.line(df, x="day", y="count",
                  title="Prediction Activity Over Time",
                  markers=True)
    return fig


def ia_trend_chart(student_id: str):
    """Bar chart of a student's IA1 → IA2 → IA3 scores."""
    rows = query("""
        SELECT ia1_score, ia2_score, ia3_score
        FROM academic_records WHERE student_id = ?
        ORDER BY created_at DESC LIMIT 1
    """, (student_id,))
    if not rows:
        return None
    r = rows[0]
    df = pd.DataFrame({
        "Test": ["IA1", "IA2", "IA3"],
        "Score": [r["ia1_score"] or 0, r["ia2_score"] or 0, r["ia3_score"] or 0],
    })

    fig = px.bar(df, x="Test", y="Score",
                 title=f"IA Performance per Test — {student_id}",
                 text="Score",
                 color="Test",
                 color_discrete_sequence=["#3498db", "#f39c12", "#e74c3c"])

    fig.update_traces(texttemplate="%{text}", textposition="outside")
    fig.update_layout(yaxis_range=[0, 55])
    return fig


def model_comparison_chart():
    """Bar chart comparing ML vs DL model metrics."""
    rows = query("""
        SELECT model_name, model_type, evaluation_metrics
        FROM model_registry ORDER BY model_type, model_name
    """)
    if not rows:
        return None
    data = []
    for r in rows:
        if not r["evaluation_metrics"]:
            continue
        m = json.loads(r["evaluation_metrics"])
        data.append({
            "Model": r["model_name"],
            "Type": r["model_type"].upper(),
            "MAE": m.get("MAE", 0),
            "R²": m.get("R2", 0),
        })
    if not data:
        return None
    df = pd.DataFrame(data)

    fig = go.Figure()
    fig.add_trace(go.Bar(name="MAE", x=df["Model"], y=df["MAE"],
                         marker_color="#e74c3c"))
    fig.add_trace(go.Bar(name="R² (×100)", x=df["Model"],
                         y=df["R²"] * 100, marker_color="#2ecc71"))
    fig.update_layout(title="ML vs DL Model Comparison",
                      barmode="group")
    return fig