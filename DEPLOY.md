# Free deployment guide — Streamlit Community Cloud

This app deploys for free on **Streamlit Community Cloud**, and every
`git push` to the connected branch **automatically rebuilds and redeploys**
the live app. That is the "changes visible in deployment" behaviour you want.

Free tier: 1 GB RAM, apps sleep after inactivity and wake on the next visit.

---

## 1. Put the project on GitHub

```bash
cd "C:\Users\DELL\Desktop\edu-predict-v2"
git init
git add .
git commit -m "Deploy-ready EduPredict"
git branch -M main
git remote add origin https://github.com/<your-username>/edu-predict.git
git push -u origin main
```

> Use a **private** repository: `database/student_perf.db` contains student
> names, IDs and marks. Streamlit Community Cloud supports private repos.

Run `python deploy_check.py` first — it lists any runtime file that is missing.

---

## 2. Create the app

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. Click **Create app** → **Deploy a public app from GitHub** (private works too).
3. Fill in:
   - **Repository:** `<your-username>/edu-predict`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Open **Advanced settings**:
   - **Python version:** `3.11` (matches the pinned wheels)
   - **Secrets:** paste the values below
5. Click **Deploy**.

### Secrets

```toml
ADMIN_USERNAME = "admin"
ADMIN_EMAIL = "you@college.edu"
ADMIN_PASSWORD = "your-strong-password"
SESSION_TIMEOUT_MINUTES = "60"
APP_SECRET_KEY = "your-random-secret"
DB_PATH = "database/student_perf.db"
```

`app.py` copies these into environment variables before the database module
loads, so no `.env` file is needed in the cloud.

---

## 3. How updates work

```bash
# edit any file locally, then:
git add .
git commit -m "Fix SHAP labels"
git push
```

Streamlit Cloud detects the push and rebuilds automatically (usually under a
minute). You can also click **Reboot app** in the dashboard to force a rebuild.

---

## 4. Important: data persistence

Streamlit Community Cloud has an **ephemeral filesystem**. The app starts from
the files committed in Git, and anything written at runtime is lost when the
app is rebuilt (i.e. on every code push) or restarted.

What resets:
- new users / signups
- new predictions in `predictions`
- activity log rows
- new `data/subject_records/*.json` files

That is fine for a demo. If you need real persistence for free, connect a free
managed database and keep the app itself on Streamlit Cloud:

| Option | Free tier | Effort |
|---|---|---|
| **Supabase** (Postgres) | 500 MB | Rewrite `database/database.py` to use `psycopg` |
| **Neon** (Postgres) | 0.5 GB | Same as above |
| **Turso** (SQLite-compatible) | 500 MB, 1B reads | Replace `sqlite3` with `libsql-client` |

Until then, keep the committed `database/student_perf.db` as the seed state.

---

## 5. Troubleshooting

- **Build fails on `lime`/`shap`** → make sure Python is set to **3.11**.
- **"Dynamic Model not found"** → `saved_models/dynamic_xgb_model.pkl` was not
  committed. Check `.gitignore` and run `git ls-files saved_models`.
- **"No such table"** → `database/student_perf.db` was not committed, or
  `DB_PATH` points somewhere else. Run `python deploy_check.py`.
- **App is slow to wake** → normal on the free tier; the first visit after
  sleep takes a few seconds.
- **Admin "Retrain" button fails** → expected: TensorFlow is not installed on
  the free deployment (it is only needed locally, see `requirements-train.txt`).

---

## Alternative: Hugging Face Spaces

If you ever hit the 1 GB RAM limit, Hugging Face Spaces also offers a free
Streamlit SDK with more RAM. Create a Space → SDK: Streamlit → push the same
repo. It rebuilds on every push too.
