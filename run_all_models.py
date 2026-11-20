"""Print model comparison table — paste into your report."""
import json
from database.database import query

rows = query("""
    SELECT model_name, model_type, active_status, evaluation_metrics
    FROM model_registry
    ORDER BY model_type, model_name
""")

if not rows:
    print("No models in registry yet.")
    print("Run these first:")
    print("  python -m machine_learning.train_ml")
    print("  python -m deep_learning.train_ann")
    raise SystemExit(1)

header = f"{'Model':20s} {'Type':6s} {'Active':7s} {'MAE':>8s} {'RMSE':>8s} {'R2':>8s}"
print(header)
print("-" * len(header))
for r in rows:
    metrics = json.loads(r["evaluation_metrics"]) if r["evaluation_metrics"] else {}
    active = "YES" if r["active_status"] else ""
    print(
        f"{r['model_name']:20s} "
        f"{r['model_type']:6s} "
        f"{active:7s} "
        f"{metrics.get('MAE', 0):8.2f} "
        f"{metrics.get('RMSE', 0):8.2f} "
        f"{metrics.get('R2', 0):8.3f}"
    )