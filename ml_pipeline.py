"""
Ransomware detection pipeline (AdaBoost + Deep Neural Network).

This module is the cleaned-up, reusable version of the original training
script (untitled0.py).  Same steps, same defaults:

  load CSV -> fill missing -> TF-IDF for text columns -> SelectKBest(chi2, k)
  -> stratified 80/20 split -> AdaBoost + DNN -> accuracy/precision/recall/F1/FPR

What is new compared with the original code:
  * the fitted preprocessing (TF-IDF + feature selector) is SAVED, so a single
    file / a CSV of files can really be classified later;
  * everything is callable from the web app (training runs in a thread);
  * the DNN uses TensorFlow/Keras when installed, otherwise scikit-learn MLP.
"""
from __future__ import annotations

import json
import os
import threading
import time
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
DEFAULT_DATASET = os.path.join(DATA_DIR, "data_file.csv")
UPLOADED_DATASET = os.path.join(DATA_DIR, "uploaded_dataset.csv")
BUNDLE_PATH = os.path.join(MODEL_DIR, "bundle.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
KERAS_PATH = os.path.join(MODEL_DIR, "dnn.keras")

DEFAULT_TARGET = "Benign"
RANDOM_STATE = 42


# --------------------------------------------------------------------------
# Optional TensorFlow
# --------------------------------------------------------------------------
_keras_cache = {}


def get_keras():
    """Return the keras module if TensorFlow is installed, else None."""
    if "k" not in _keras_cache:
        try:
            from tensorflow import keras  # type: ignore
            _keras_cache["k"] = keras
        except Exception:
            _keras_cache["k"] = None
    return _keras_cache["k"]


# --------------------------------------------------------------------------
# Preprocessing (fit once, reuse for prediction)
# --------------------------------------------------------------------------
class Preprocessor:
    def __init__(self, target: str = DEFAULT_TARGET, k: int = 20, tfidf_features: int = 100):
        self.target = target
        self.k = k
        self.tfidf_features = tfidf_features

    # ---- fit -------------------------------------------------------------
    def fit(self, df: pd.DataFrame):
        df = df.dropna(subset=[self.target]).reset_index(drop=True)
        raw = df[self.target]
        classes = sorted(raw.unique().tolist(), key=str)
        if len(classes) != 2:
            raise ValueError(
                f"Target column '{self.target}' must have exactly 2 classes "
                f"(found {len(classes)}).")
        self.classes_ = classes
        y = (raw == classes[1]).astype(int).to_numpy()

        X = df.drop(columns=[self.target])
        self.num_cols = [c for c in X.columns
                         if pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c])]
        self.text_cols = [c for c in X.columns if c not in self.num_cols]
        if not self.num_cols and not self.text_cols:
            raise ValueError("The dataset has no feature columns.")

        # numeric: median fill, shift so chi2 gets non-negative values
        num = X[self.num_cols].apply(pd.to_numeric, errors="coerce").astype(float)
        self.medians = num.median().fillna(0.0).to_dict()
        num = num.fillna(self.medians)
        self.shift = {c: float(max(0.0, -num[c].min())) for c in self.num_cols}
        num = num + pd.Series(self.shift)

        # text: one TF-IDF per column
        self.vectorizers, blocks, names = {}, [], list(self.num_cols)
        for c in self.text_cols:
            vec = TfidfVectorizer(max_features=self.tfidf_features)
            try:
                m = vec.fit_transform(X[c].fillna("").astype(str))
            except ValueError:          # column with no usable tokens
                continue
            self.vectorizers[c] = vec
            blocks.append(m)
            names += [f"{c}: '{t}'" for t in vec.get_feature_names_out()]
        self.text_cols = list(self.vectorizers)
        self.feature_names = names

        full = hstack([csr_matrix(num.values)] + blocks).tocsr()
        k = max(1, min(int(self.k), full.shape[1]))
        self.k = k
        self.selector = SelectKBest(score_func=chi2, k=k).fit(full, y)
        self.selected_idx = self.selector.get_support(indices=True)
        self.selected_names = [names[i] for i in self.selected_idx]

        # stats used by the prediction form
        self.stats = {}
        for c in self.num_cols:
            col = X[c].apply(pd.to_numeric, errors="coerce")
            self.stats[c] = {
                "min": float(col.min()), "max": float(col.max()),
                "median": float(col.median()),
                "median_class0": float(col[y == 0].median()),
                "median_class1": float(col[y == 1].median()),
            }
        self._n_num = len(self.num_cols)
        self._text_ranges, pos = {}, self._n_num
        for c in self.text_cols:
            n = len(self.vectorizers[c].get_feature_names_out())
            self._text_ranges[c] = (pos, pos + n)
            pos += n
        self._full_shape = full.shape
        return self, full[:, self.selected_idx], y

    # ---- transform ---------------------------------------------------------
    def transform(self, df: pd.DataFrame):
        n = len(df)
        num = pd.DataFrame(index=df.index)
        for c in self.num_cols:
            col = pd.to_numeric(df[c], errors="coerce") if c in df else pd.Series(np.nan, index=df.index)
            num[c] = col.fillna(self.medians[c]).astype(float)
        num = (num + pd.Series(self.shift)).clip(lower=0)
        blocks = []
        for c in self.text_cols:
            s = df[c].fillna("").astype(str) if c in df else pd.Series([""] * n, index=df.index)
            blocks.append(self.vectorizers[c].transform(s))
        full = hstack([csr_matrix(num.values)] + blocks).tocsr()
        return full[:, self.selected_idx]

    # ---- helpers for the UI ------------------------------------------------
    def used_columns(self):
        sel = set(int(i) for i in self.selected_idx)
        used_num = [c for i, c in enumerate(self.num_cols) if i in sel]
        used_text = [c for c, (a, b) in self._text_ranges.items() if any(i in sel for i in range(a, b))]
        return used_num, used_text

    def labels(self):
        c0, c1 = self.classes_
        if str(self.target).lower() == "benign":
            return {"0": "Malicious", "1": "Benign"}
        return {"0": str(c0), "1": str(c1)}

    def schema(self):
        used_num, used_text = self.used_columns()
        fields = []
        for c in used_num:
            fields.append({"name": c, "kind": "number", **self.stats[c]})
        for c in used_text:
            fields.append({"name": c, "kind": "text"})
        return {"target": self.target, "labels": self.labels(), "fields": fields,
                "k": self.k, "selected": self.selected_names}


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def _make_adaboost():
    try:
        return AdaBoostClassifier(random_state=RANDOM_STATE, n_estimators=50, algorithm="SAMME")
    except TypeError:                       # newer scikit-learn removed the arg
        return AdaBoostClassifier(random_state=RANDOM_STATE, n_estimators=50)


def _scores(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


class DNN:
    """Thin wrapper so Keras and scikit-learn MLP expose the same API."""

    def __init__(self, engine, model, scaler):
        self.engine, self.model, self.scaler = engine, model, scaler

    def predict_proba(self, X):
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        if self.engine == "keras":
            return np.asarray(self.model.predict(X, verbose=0)).ravel()
        return self.model.predict_proba(X)[:, 1]


def _train_dnn(Xtr, ytr, epochs: int, scale: bool):
    Xtr = Xtr.toarray()
    scaler = StandardScaler().fit(Xtr) if scale else None
    Xin = scaler.transform(Xtr) if scaler is not None else Xtr
    keras = get_keras()
    if keras is not None:
        L = keras.layers
        model = keras.Sequential([
            keras.Input(shape=(Xin.shape[1],)),
            L.Dense(256, activation="relu"), L.BatchNormalization(), L.Dropout(0.3),
            L.Dense(128, activation="relu"), L.BatchNormalization(), L.Dropout(0.2),
            L.Dense(64, activation="relu"), L.Dropout(0.1),
            L.Dense(1, activation="sigmoid"),
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        model.fit(Xin, ytr, epochs=epochs, batch_size=32, validation_split=0.1,
                  callbacks=[stop], verbose=0)
        return DNN("keras", model, scaler)
    model = MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation="relu", solver="adam",
                          batch_size=32, early_stopping=True, validation_fraction=0.1,
                          n_iter_no_change=5, max_iter=epochs, random_state=RANDOM_STATE)
    model.fit(Xin, ytr)
    return DNN("sklearn-mlp", model, scaler)


# --------------------------------------------------------------------------
# Global state: dataset, trained bundle, background training job
# --------------------------------------------------------------------------
_lock = threading.RLock()
STATE = {"df": None, "dataset_name": None, "bundle": None, "metrics": None}
JOB = {"running": False, "step": 0, "steps": [], "error": None, "done": False}

STEPS = ["Prepare data", "Select features", "Split train / test", "Train AdaBoost",
         "Train neural network", "Save models"]


def load_dataset(path: str = DEFAULT_DATASET, name: str | None = None):
    df = pd.read_csv(path)
    if name is None:
        name = ("data_file.csv (bundled)" if os.path.abspath(path) == os.path.abspath(DEFAULT_DATASET)
                else os.path.basename(path))
    with _lock:
        STATE["df"] = df
        STATE["dataset_name"] = name
    return df


def reset_dataset():
    if os.path.exists(UPLOADED_DATASET):
        os.remove(UPLOADED_DATASET)
    return load_dataset(DEFAULT_DATASET, "data_file.csv (bundled)")


def train_all(df, target, k, test_size, scale_dnn, epochs, dataset_name, progress=lambda i: None):
    t0 = time.time()
    progress(0)
    pre = Preprocessor(target=target, k=k)
    progress(1)
    pre, Xs, y = pre.fit(df)
    progress(2)
    idx = np.arange(len(y))
    Xtr, Xte, ytr, yte, _, test_idx = train_test_split(
        Xs, y, idx, test_size=test_size, random_state=RANDOM_STATE, stratify=y)
    progress(3)
    ada = _make_adaboost().fit(Xtr, ytr)
    ada_m = _scores(yte, ada.predict(Xte))
    progress(4)
    dnn = _train_dnn(Xtr, ytr, epochs, scale_dnn)
    dnn_m = _scores(yte, (dnn.predict_proba(Xte) >= 0.5).astype(int))
    progress(5)

    importances = sorted(zip(pre.selected_names, ada.feature_importances_.tolist()),
                         key=lambda t: -t[1])
    metrics = {
        "adaboost": ada_m, "dnn": dnn_m,
        "dnn_engine": dnn.engine, "dnn_scaled": bool(scale_dnn),
        "importances": [{"feature": n, "value": v} for n, v in importances],
        "params": {"target": target, "k": pre.k, "test_size": test_size, "epochs": epochs},
        "dataset": {"name": dataset_name, "rows": int(len(y)),
                    "class0": int((y == 0).sum()), "class1": int((y == 1).sum()),
                    "train_rows": int(len(ytr)), "test_rows": int(len(yte))},
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seconds": round(time.time() - t0, 1),
    }
    bundle = {"pre": pre, "ada": ada, "dnn": dnn, "test_idx": test_idx, "schema": pre.schema()}
    save_bundle(bundle, metrics)
    with _lock:
        STATE["bundle"], STATE["metrics"] = bundle, metrics
    return metrics


def save_bundle(bundle, metrics):
    os.makedirs(MODEL_DIR, exist_ok=True)
    dnn = bundle["dnn"]
    to_dump = dict(bundle)
    if dnn.engine == "keras":
        dnn.model.save(KERAS_PATH)
        to_dump["dnn"] = DNN("keras", None, dnn.scaler)
    joblib.dump(to_dump, BUNDLE_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)


def load_saved():
    """Load models saved by a previous run.  Returns True on success."""
    try:
        bundle = joblib.load(BUNDLE_PATH)
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
        if bundle["dnn"].engine == "keras":
            keras = get_keras()
            if keras is None:
                return False
            bundle["dnn"].model = keras.models.load_model(KERAS_PATH)
    except Exception:
        return False
    with _lock:
        STATE["bundle"], STATE["metrics"] = bundle, metrics
    return True


def start_training(target, k, test_size, scale_dnn, epochs):
    """Run training in a background thread; poll with job_status()."""
    with _lock:
        if JOB["running"]:
            raise RuntimeError("Training is already running.")
        df, name = STATE["df"], STATE["dataset_name"]
        if df is None:
            raise RuntimeError("No dataset loaded.")
        if target not in df.columns:
            raise ValueError(f"Column '{target}' not found in the dataset.")
        JOB.update(running=True, step=0, steps=STEPS, error=None, done=False)

    def run():
        try:
            train_all(df.copy(), target, k, test_size, scale_dnn, epochs, name,
                      progress=lambda i: JOB.update(step=i))
            JOB.update(done=True, step=len(STEPS))
        except Exception as e:                      # surfaced in the UI
            JOB.update(error=str(e))
        finally:
            JOB.update(running=False)

    threading.Thread(target=run, daemon=True).start()


def job_status():
    return dict(JOB)


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
def predict(df: pd.DataFrame, model: str = "adaboost"):
    """Return (labels 0/1, probability of class 1) for every row of df."""
    with _lock:
        b = STATE["bundle"]
    if b is None:
        raise RuntimeError("No trained model yet. Train the models first.")
    X = b["pre"].transform(df)
    if model == "dnn":
        p = b["dnn"].predict_proba(X)
    else:
        p = b["ada"].predict_proba(X)[:, 1]
    return (p >= 0.5).astype(int), p


def sample_row(label: int):
    """A random held-out test row of the requested class (for demos)."""
    with _lock:
        b, df = STATE["bundle"], STATE["df"]
    if b is None or df is None:
        raise RuntimeError("No trained model yet.")
    pre = b["pre"]
    d = df.dropna(subset=[pre.target]).reset_index(drop=True)
    y = (d[pre.target] == pre.classes_[1]).astype(int).to_numpy()
    pool = [int(i) for i in b["test_idx"] if i < len(d) and y[i] == label]
    if not pool:
        raise RuntimeError("No held-out sample of that class.")
    row = d.iloc[int(np.random.choice(pool))]
    used_num, used_text = pre.used_columns()
    return {c: (row[c].item() if hasattr(row[c], "item") else row[c]) for c in used_num + used_text}


def dataset_summary(df: pd.DataFrame):
    miss = df.isnull().sum()
    num = df.select_dtypes("number")
    desc = num.describe().T[["mean", "std", "min", "max"]].round(2) if not num.empty else pd.DataFrame()
    return {
        "rows": int(df.shape[0]), "cols": int(df.shape[1]),
        "columns": [{"name": c, "dtype": ("number" if pd.api.types.is_numeric_dtype(df[c]) else "text"),
                     "missing": int(miss[c]), "unique": int(df[c].nunique())} for c in df.columns],
        "head": json.loads(df.head(8).to_json(orient="split")),
        "stats": [{"name": i, **r} for i, r in desc.to_dict("index").items()],
        "missing_total": int(miss.sum()),
    }


def init():
    """Called once at server start."""
    if os.path.exists(UPLOADED_DATASET):
        load_dataset(UPLOADED_DATASET, "uploaded_dataset.csv")
    else:
        load_dataset(DEFAULT_DATASET, "data_file.csv (bundled)")
    return load_saved()
