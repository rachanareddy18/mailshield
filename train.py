"""Train and evaluate the spam models.   py train.py          (full tuning, a few minutes)
                                          py train.py --fast   (quick retrain, used by the app)"""
import json
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB

from features import clean_text

DATA, MODELS = Path("data"), Path("models")
LABEL_COLS = ["label", "spam", "category", "class", "target", "v1", "is_spam"]
TEXT_COLS = ["text", "message", "email", "body", "content", "v2", "email_text"]
FAST = "--fast" in sys.argv


def to_bin(v):
    s = str(v).strip().lower()
    if s in {"spam", "1", "1.0", "true", "junk"}:
        return 1
    if s in {"ham", "0", "0.0", "false", "not spam", "legitimate", "safe"}:
        return 0
    return None


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1")
    df.columns = [c.strip().lower() for c in df.columns]
    lab = next((c for c in LABEL_COLS if c in df.columns), None)
    txt = next((c for c in TEXT_COLS if c in df.columns), None)
    if lab is None or txt is None:
        sys.exit(f"{path.name}: could not find label/text columns. Found: {list(df.columns)}")
    text = df[txt].astype(str)
    if "subject" in df.columns and txt != "subject":
        text = df["subject"].astype(str) + " . " + text
    out = pd.DataFrame({"text": text, "label": df[lab].map(to_bin)}).dropna()
    out["label"] = out["label"].astype(int)
    return out.drop_duplicates(subset="text")


def metrics(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    d = lambda a, b: float(a / b) if b else 0.0
    prec, rec = d(tp, tp + fp), d(tp, tp + fn)
    return {"accuracy": d(tp + tn, tp + tn + fp + fn), "precision": prec, "recall": rec,
            "specificity": d(tn, tn + fp), "f1": d(2 * prec * rec, prec + rec),
            "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn)}


def main():
    t0 = time.time()
    base = [p for p in (DATA / "enron.csv", DATA / "emails.csv") if p.exists()]
    if not base:
        sys.exit("No base dataset. Run:  py get_data.py   (or put an email CSV at data/emails.csv)")
    files = base[:1] + sorted(DATA.glob("extra_*.csv")) + [p for p in (DATA / "campus_train.csv", DATA / "feedback.csv") if p.exists()]
    parts = []
    for f in files:
        d = load_dataset(f)
        d["source"] = f.stem
        parts.append(d)
        print(f"  {f.name:22s} {len(d):6d} emails  (spam {int(d.label.sum())})")
    df = pd.concat(parts).drop_duplicates(subset="text").reset_index(drop=True)
    print(f"Total: {len(df)} emails | spam {int(df.label.sum())} | ham {int((1 - df.label).sum())}")
    print("Cleaning text ...")
    df["clean"] = df.text.map(clean_text)
    df = df[df.clean.str.len() > 0].reset_index(drop=True)

    train, test = train_test_split(df, test_size=0.2, stratify=df.label, random_state=42)
    ytr, yte = train.label.to_numpy(), test.label.to_numpy()

    ood = pd.read_csv(DATA / "campus_test.csv") if (DATA / "campus_test.csv").exists() else None
    if ood is not None:
        ood["y"] = ood.label.map(to_bin)
        ood = ood.dropna(subset=["y"]).reset_index(drop=True)
        ood["clean"] = ood.text.map(clean_text)

    vec_cfgs = {"Unigrams + bigrams": (1, 2)} if FAST else {"Unigrams": (1, 1), "Unigrams + bigrams": (1, 2)}
    grids = {"Multinomial Naive Bayes": (MultinomialNB(), {"alpha": [0.05, 0.1, 0.5] if FAST else [0.01, 0.05, 0.1, 0.5, 1.0]}),
             "Logistic Regression": (LogisticRegression(max_iter=3000, class_weight="balanced"),
                                     {"C": [1, 10] if FAST else [0.5, 1, 5, 10, 20]})}
    cv = StratifiedKFold(3 if FAST else 5, shuffle=True, random_state=42)

    tuning, fitted = [], {}
    for vname, ngram in vec_cfgs.items():
        vec = TfidfVectorizer(stop_words="english", ngram_range=ngram, min_df=2, max_df=0.9, sublinear_tf=True, max_features=60000)
        Xtr = vec.fit_transform(train.clean)
        res = {}
        for mname, (est, grid) in grids.items():
            print(f"Tuning {mname} with {vname} ({cv.get_n_splits()}-fold CV) ...")
            gs = GridSearchCV(est, grid, cv=cv, scoring="f1", n_jobs=-1).fit(Xtr, ytr)
            res[mname] = gs
            tuning.append({"Features": vname, "Model": mname, "Best settings": json.dumps(gs.best_params_),
                           "CV F1": round(float(gs.best_score_), 4)})
        fitted[vname] = (vec, Xtr, res, float(np.mean([g.best_score_ for g in res.values()])))
    best_v = max(fitted, key=lambda k: fitted[k][3])
    vec, Xtr, res, _ = fitted[best_v]
    print("Best feature set:", best_v)
    Xte = vec.transform(test.clean)
    Xood = vec.transform(ood.clean) if ood is not None else None

    models, meta_models, proba, ood_proba, errors = {}, {}, {}, {}, []
    for name, gs in res.items():
        m = gs.best_estimator_
        models[name] = m
        p = m.predict_proba(Xte)[:, 1]
        proba[name] = p
        info = {"best_params": gs.best_params_, "train_accuracy": float((m.predict(Xtr) == ytr).mean()),
                "cv_f1": float(gs.best_score_), "test": metrics(yte, (p >= 0.5).astype(int))}
        info["test_by_source"] = {s: float(((p[test.source.values == s] >= 0.5).astype(int) == yte[test.source.values == s]).mean())
                                  for s in test.source.unique()}
        if Xood is not None:
            po = m.predict_proba(Xood)[:, 1]
            ood_proba[name] = po
            info["campus_test"] = metrics(ood.y.astype(int).to_numpy(), (po >= 0.5).astype(int))
        meta_models[name] = info
        print(f"{name}: test acc {info['test']['accuracy']:.4f} F1 {info['test']['f1']:.4f}"
              + (f" | campus_test acc {info['campus_test']['accuracy']:.4f}" if Xood is not None else ""))

    lr = "Logistic Regression"
    wrong = (proba[lr] >= 0.5).astype(int) != yte
    e1 = pd.DataFrame({"Email": test.text.str[:180].values[wrong], "Actual": np.where(yte[wrong] == 1, "spam", "ham"),
                       "NB spam %": (proba["Multinomial Naive Bayes"][wrong] * 100).round(1),
                       "LR spam %": (proba[lr][wrong] * 100).round(1), "Source": test.source.values[wrong]})
    err = [e1]
    if ood is not None:
        w2 = (ood_proba[lr] >= 0.5).astype(int) != ood.y.astype(int).to_numpy()
        err.append(pd.DataFrame({"Email": ood.text.str[:180].values[w2], "Actual": np.where(ood.y.values[w2] == 1, "spam", "ham"),
                                 "NB spam %": (ood_proba["Multinomial Naive Bayes"][w2] * 100).round(1),
                                 "LR spam %": (ood_proba[lr][w2] * 100).round(1), "Source": "campus_test"}))
    MODELS.mkdir(exist_ok=True)
    pd.concat(err).to_csv(MODELS / "errors.csv", index=False)

    meta = {"n_emails": int(len(df)), "n_spam": int(df.label.sum()), "sources": df.source.value_counts().to_dict(),
            "features": best_v, "tuning": tuning, "models": meta_models, "fast": FAST, "seconds": round(time.time() - t0)}
    if (MODELS / "metrics.json").exists():
        shutil.copy(MODELS / "metrics.json", MODELS / "metrics_prev.json")
    (MODELS / "metrics.json").write_text(json.dumps(meta, indent=2))
    joblib.dump({"vectorizer": vec, "models": models}, MODELS / "bundle.joblib")
    joblib.dump({"y_test": yte, "proba": proba, "campus_y": None if ood is None else ood.y.astype(int).to_numpy(),
                 "campus_proba": ood_proba}, MODELS / "eval.joblib")
    print(f"Done in {time.time() - t0:.0f}s. Now run:  py -m streamlit run app.py")


if __name__ == "__main__":
    main()
