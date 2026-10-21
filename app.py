"""
Ransomware Detection - web application (Flask).

Run:   python app.py        ->  http://127.0.0.1:5000
"""
import hashlib
import io
import os
import re
import secrets
import sqlite3
import time
from functools import wraps

import pandas as pd
from flask import (Flask, abort, jsonify, redirect, render_template, request,
                   session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash

import ml_pipeline as ml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "users.db")
KEY_PATH = os.path.join(BASE_DIR, "instance", "secret.key")

app = Flask(__name__)
app.config.update(
    MAX_CONTENT_LENGTH=60 * 1024 * 1024,        # uploads up to 60 MB
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)


def _secret_key():
    if os.environ.get("SECRET_KEY"):
        return os.environ["SECRET_KEY"]
    os.makedirs(os.path.dirname(KEY_PATH), exist_ok=True)
    if not os.path.exists(KEY_PATH):
        with open(KEY_PATH, "w") as f:
            f.write(secrets.token_hex(32))
    return open(KEY_PATH).read().strip()


app.secret_key = _secret_key()


# --------------------------------------------------------------------------
# Users (same `users` table as the original project)
# --------------------------------------------------------------------------
def db():
    return sqlite3.connect(DB_PATH)


def init_db():
    with db() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS users
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      username TEXT UNIQUE,
                      password TEXT)""")


def legacy_hash(password):               # hash used by the original Streamlit app
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(stored, password):
    if stored.startswith(("scrypt:", "pbkdf2:")):
        return check_password_hash(stored, password)
    return secrets.compare_digest(stored, legacy_hash(password))


_attempts = {}        # (ip, username) -> [timestamps]  (simple brute-force guard)


def too_many_attempts(key):
    now = time.time()
    _attempts[key] = [t for t in _attempts.get(key, []) if now - t < 300]
    return len(_attempts[key]) >= 8


# --------------------------------------------------------------------------
# Auth helpers + CSRF
# --------------------------------------------------------------------------
def csrf_token():
    if "csrf" not in session:
        session["csrf"] = secrets.token_hex(16)
    return session["csrf"]


app.jinja_env.globals["csrf_token"] = csrf_token


@app.before_request
def check_csrf():
    if request.method in ("POST", "PUT", "DELETE") and request.path.startswith("/api/"):
        sent = request.headers.get("X-CSRF-Token", "")
        if not sent or not secrets.compare_digest(sent, session.get("csrf", "")):
            return jsonify(error="Your session expired. Reload the page and try again."), 400


def login_required(view):
    @wraps(view)
    def wrapper(*a, **kw):
        if "user" not in session:
            if request.path.startswith("/api/"):
                return jsonify(error="Please sign in first."), 401
            return redirect(url_for("login_page"))
        return view(*a, **kw)
    return wrapper


@app.errorhandler(413)
def too_big(_):
    return jsonify(error="That file is larger than 60 MB."), 413


@app.after_request
def headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Cache-Control"] = "no-store" if request.path.startswith("/api/") else resp.headers.get("Cache-Control", "no-cache")
    return resp


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def page(template, active):
    return render_template(template, active=active, user=session.get("user"))


@app.route("/")
@login_required
def overview():
    return page("overview.html", "overview")


@app.route("/dataset")
@login_required
def dataset_page():
    return page("dataset.html", "dataset")


@app.route("/train")
@login_required
def train_page():
    return page("train.html", "train")


@app.route("/predict")
@login_required
def predict_page():
    return page("predict.html", "predict")


@app.route("/results")
@login_required
def results_page():
    return page("results.html", "results")


@app.route("/login")
def login_page():
    if "user" in session:
        return redirect(url_for("overview"))
    return render_template("login.html")


@app.route("/register")
def register_page():
    if "user" in session:
        return redirect(url_for("overview"))
    return render_template("register.html")


# --------------------------------------------------------------------------
# Auth API
# --------------------------------------------------------------------------
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")


@app.post("/api/register")
def api_register():
    d = request.get_json(silent=True) or {}
    username, password = (d.get("username") or "").strip(), d.get("password") or ""
    if not USERNAME_RE.match(username):
        return jsonify(error="Username must be 3-32 characters: letters, numbers, . _ -"), 400
    if len(password) < 6:
        return jsonify(error="Password must be at least 6 characters."), 400
    try:
        with db() as c:
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                      (username, generate_password_hash(password)))
    except sqlite3.IntegrityError:
        return jsonify(error="That username is taken. Choose another one."), 409
    return jsonify(ok=True)


@app.post("/api/login")
def api_login():
    d = request.get_json(silent=True) or {}
    username, password = (d.get("username") or "").strip(), d.get("password") or ""
    key = (request.remote_addr, username.lower())
    if too_many_attempts(key):
        return jsonify(error="Too many attempts. Wait a few minutes and try again."), 429
    with db() as c:
        row = c.execute("SELECT password FROM users WHERE username = ?", (username,)).fetchone()
    if not row or not verify_password(row[0], password):
        _attempts.setdefault(key, []).append(time.time())
        return jsonify(error="Wrong username or password."), 401
    if not row[0].startswith(("scrypt:", "pbkdf2:")):          # upgrade old SHA-256 hash
        with db() as c:
            c.execute("UPDATE users SET password = ? WHERE username = ?",
                      (generate_password_hash(password), username))
    session.clear()
    session["user"] = username
    csrf_token()
    return jsonify(ok=True)


@app.post("/api/logout")
def api_logout():
    session.clear()
    return jsonify(ok=True)


# --------------------------------------------------------------------------
# Data / model API
# --------------------------------------------------------------------------
@app.get("/api/status")
@login_required
def api_status():
    m = ml.STATE["metrics"]
    df = ml.STATE["df"]
    return jsonify(
        dataset={"name": ml.STATE["dataset_name"], "rows": int(len(df)), "cols": int(df.shape[1])},
        trained=m is not None,
        metrics=m,
        stale=bool(m and m["dataset"]["name"] != ml.STATE["dataset_name"]),
        keras=ml.get_keras() is not None,
        job=ml.job_status(),
    )


@app.get("/api/dataset")
@login_required
def api_dataset():
    df = ml.STATE["df"]
    out = ml.dataset_summary(df)
    out["name"] = ml.STATE["dataset_name"]
    out["target_candidates"] = [c for c in df.columns
                                if df[c].nunique(dropna=True) == 2]
    out["default_target"] = ml.DEFAULT_TARGET if ml.DEFAULT_TARGET in df.columns else (
        out["target_candidates"][0] if out["target_candidates"] else None)
    if out["default_target"]:
        vc = df[out["default_target"]].value_counts().sort_index()
        out["class_counts"] = {str(k): int(v) for k, v in vc.items()}
    return jsonify(out)


@app.post("/api/dataset/upload")
@login_required
def api_upload():
    f = request.files.get("file")
    if not f or not f.filename.lower().endswith(".csv"):
        return jsonify(error="Choose a .csv file."), 400
    try:
        df = pd.read_csv(f)
    except Exception as e:
        return jsonify(error=f"Could not read that file as CSV: {e}"), 400
    if df.shape[0] < 50 or df.shape[1] < 2:
        return jsonify(error="The CSV needs at least 50 rows and 2 columns."), 400
    df.to_csv(ml.UPLOADED_DATASET, index=False)
    ml.load_dataset(ml.UPLOADED_DATASET, "uploaded_dataset.csv")
    return jsonify(ok=True)


@app.post("/api/dataset/reset")
@login_required
def api_reset():
    ml.reset_dataset()
    return jsonify(ok=True)


@app.post("/api/train")
@login_required
def api_train():
    d = request.get_json(silent=True) or {}
    try:
        target = d.get("target") or ml.DEFAULT_TARGET
        k = int(d.get("k", 20))
        test_size = float(d.get("test_size", 0.2))
        epochs = int(d.get("epochs", 30))
        if not (1 <= k <= 500 and 0.1 <= test_size <= 0.5 and 1 <= epochs <= 100):
            raise ValueError("Settings out of range.")
        ml.start_training(target, k, test_size, bool(d.get("scale_dnn", True)), epochs)
    except (ValueError, RuntimeError) as e:
        return jsonify(error=str(e)), 400
    return jsonify(ok=True)


@app.get("/api/train/status")
@login_required
def api_train_status():
    return jsonify(ml.job_status())


@app.get("/api/schema")
@login_required
def api_schema():
    b = ml.STATE["bundle"]
    if b is None:
        return jsonify(error="No trained model yet."), 404
    return jsonify(b["schema"])


@app.get("/api/sample")
@login_required
def api_sample():
    try:
        label = 1 if request.args.get("label") == "1" else 0
        return jsonify(ml.sample_row(label))
    except RuntimeError as e:
        return jsonify(error=str(e)), 404


def _model_name(v):
    return "dnn" if v == "dnn" else "adaboost"


@app.post("/api/predict")
@login_required
def api_predict():
    d = request.get_json(silent=True) or {}
    b = ml.STATE["bundle"]
    if b is None:
        return jsonify(error="Train the models first."), 400
    values = d.get("values") or {}
    row = {}
    for f in b["schema"]["fields"]:
        v = values.get(f["name"], "")
        if f["kind"] == "number":
            try:
                row[f["name"]] = float(v)
            except (TypeError, ValueError):
                return jsonify(error=f"'{f['name']}' must be a number."), 400
        else:
            row[f["name"]] = str(v)
    model = _model_name(d.get("model"))
    labels, proba = ml.predict(pd.DataFrame([row]), model)
    names = b["schema"]["labels"]
    return jsonify(label=int(labels[0]), verdict=names[str(int(labels[0]))],
                   p_class1=float(proba[0]), model=model, labels=names)


@app.post("/api/predict/batch")
@login_required
def api_predict_batch():
    f = request.files.get("file")
    b = ml.STATE["bundle"]
    if b is None:
        return jsonify(error="Train the models first."), 400
    if not f or not f.filename.lower().endswith(".csv"):
        return jsonify(error="Choose a .csv file."), 400
    try:
        df = pd.read_csv(f)
    except Exception as e:
        return jsonify(error=f"Could not read that file as CSV: {e}"), 400
    needed = [x["name"] for x in b["schema"]["fields"] if x["kind"] == "number"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return jsonify(error="Missing columns: " + ", ".join(missing)), 400
    if len(df) > 20000:
        return jsonify(error="Please upload at most 20,000 rows at a time."), 400
    model = _model_name(request.form.get("model"))
    labels, proba = ml.predict(df, model)
    names = b["schema"]["labels"]
    text_cols = [x["name"] for x in b["schema"]["fields"] if x["kind"] == "text"]
    name_col = text_cols[0] if text_cols and text_cols[0] in df.columns else None
    rows = [{"row": i + 1,
             "name": (str(df[name_col].iloc[i]) if name_col else ""),
             "verdict": names[str(int(l))],
             "p_class1": round(float(p), 4)}
            for i, (l, p) in enumerate(zip(labels, proba))]
    counts = {names["0"]: int((labels == 0).sum()), names["1"]: int((labels == 1).sum())}
    return jsonify(rows=rows, counts=counts, labels=names)


if __name__ == "__main__":
    init_db()
    print("Loading data and models ...")
    if not ml.init():
        print("No saved models found (or they were made with another library version).")
        print("Open the 'Train models' page in the app, or run:  python train_model.py")
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
