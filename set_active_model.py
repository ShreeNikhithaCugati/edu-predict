from database.database import execute, query

# Reset all models to inactive
execute("UPDATE model_registry SET active_status = 0")

# Activate XGBoost
execute("UPDATE model_registry SET active_status = 1 WHERE model_name = 'XGBoost'")

# Verify
rows = query("SELECT model_name, active_status FROM model_registry ORDER BY model_name")
print("\nCurrent Model Registry:\n")
for r in rows:
    marker = "✓ ACTIVE" if r["active_status"] else "  inactive"
    print(f"  {marker:10s}  {r['model_name']}")

print("\nDone. Refresh the browser to see XGBoost as Active.")