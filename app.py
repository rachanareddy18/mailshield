"""MailShield - explainable email spam triage.   Run:  py -m streamlit run app.py"""
import html
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from features import (MONEY, SENSITIVE, URGENT, clean_text, dna_scores, red_flags, sender_analysis, spam_category,
                      strip_links, strip_words)
from train import LABEL_COLS, TEXT_COLS, to_bin

st.set_page_config(page_title="MailShield", page_icon="🛡️", layout="wide")
BUNDLE, EVAL, META = Path("models/bundle.joblib"), Path("models/eval.joblib"), Path("models/metrics.json")
SPAM_B, REVIEW_B, SAFE_B = "🚨 Spam", "⚠️ Review", "✅ Safe"
if not BUNDLE.exists():
    st.error("No trained model found. Run  py get_data.py  then  py train.py  first.")
    st.stop()


@st.cache_resource
def load_all():
    b = joblib.load(BUNDLE)
    return b["vectorizer"], b["models"], (joblib.load(EVAL) if EVAL.exists() else None)


vec, MODELS, evald = load_all()
FEATS = vec.get_feature_names_out()
meta = json.loads(META.read_text()) if META.exists() else {}
ss = st.session_state
for k, v in {"queue": [], "idx": 0, "running": False, "results": [], "src": None, "last": None,
             "paste": "", "sender": "", "sim_src": None, "sim_text": ""}.items():
    ss.setdefault(k, v)


# ------------------------------------------------------------------ core logic
def weights(name):
    m = MODELS[name]
    return m.feature_log_prob_[1] - m.feature_log_prob_[0] if hasattr(m, "feature_log_prob_") else m.coef_[0]


def analyze(text, sender, name, spam_th, review_th):
    X = vec.transform([clean_text(text)])
    p = float(MODELS[name].predict_proba(X)[0, 1])
    flags = red_flags(text)
    s_score, s_why = sender_analysis(sender)
    risk = min(1.0, p + min(0.25, sum(f[2] for f in flags)) + 0.15 * (s_score or 0))
    coo, w = X.tocoo(), weights(name)
    contrib = sorted(((FEATS[j], float(v * w[j])) for j, v in zip(coo.col, coo.data)), key=lambda t: -abs(t[1]))
    bucket = SPAM_B if risk >= spam_th else REVIEW_B if risk >= review_th else SAFE_B
    return {"p": p, "risk": risk, "flags": flags, "sender": (s_score, s_why), "all": contrib, "signals": contrib[:6],
            "bucket": bucket, "type": spam_category(text, flags) if risk >= review_th else "-",
            "dna": dna_scores(text, flags, s_score, p)}


def signals_text(sig):
    return ", ".join(f"{t} ({'spam+' if c > 0 else 'ham+'})" for t, c in sig[:4])


def highlight(text, contrib):
    uni = {t: c for t, c in contrib if " " not in t}
    out = []
    for tok in re.split(r"(\s+)", text):
        c = uni.get(clean_text(tok))
        safe = html.escape(tok)
        if c is None or not tok.strip():
            out.append(safe)
        else:
            col = "rgba(255,80,80,.35)" if c > 0 else "rgba(60,180,100,.35)"
            out.append(f"<span style='background:{col};border-radius:4px;padding:0 2px'>{safe}</span>")
    return "".join(out).replace("\n", "<br>")


def read_queue(df):
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    tcol = next((c for c in TEXT_COLS if c in df.columns), df.columns[0])
    scol = next((c for c in ("sender", "from") if c in df.columns), None)
    lcol = next((c for c in LABEL_COLS if c in df.columns and c != tcol), None)
    items = []
    for _, r in df.iterrows():
        y = to_bin(r[lcol]) if lcol else None
        items.append({"text": str(r[tcol]), "sender": str(r[scol]) if scol and pd.notna(r[scol]) else "", "y": y})
    return items


def counts(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    return (int(((y == 1) & (pred == 1)).sum()), int(((y == 1) & (pred == 0)).sum()),
            int(((y == 0) & (pred == 1)).sum()), int(((y == 0) & (pred == 0)).sum()))


def rates(tp, fn, fp, tn):
    d = lambda a, b: a / b if b else 0.0
    pr, rc = d(tp, tp + fp), d(tp, tp + fn)
    return {"Accuracy": d(tp + tn, tp + tn + fp + fn), "Precision": pr, "Recall": rc, "Specificity": d(tn, tn + fp),
            "F1": d(2 * pr * rc, pr + rc), "False-positive rate": d(fp, fp + tn)}


def cm_html(tp, fn, fp, tn):
    g, r = "rgba(60,180,100,.30)", "rgba(255,80,80,.30)"
    cell = "padding:14px;border:1px solid #8884"
    return (f"<table style='border-collapse:collapse;text-align:center;width:100%;max-width:480px'>"
            f"<tr><th></th><th>Predicted SPAM</th><th>Predicted HAM</th></tr>"
            f"<tr><th>Actual SPAM</th><td style='background:{g};{cell}'><b>TP</b><br>{tp}</td><td style='background:{r};{cell}'><b>FN</b><br>{fn}</td></tr>"
            f"<tr><th>Actual HAM</th><td style='background:{r};{cell}'><b>FP</b><br>{fp}</td><td style='background:{g};{cell}'><b>TN</b><br>{tn}</td></tr></table>")


def radar(dna):
    labels, vals = list(dna), list(dna.values())
    ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    vals, ang = vals + vals[:1], ang + ang[:1]
    fig = plt.figure(figsize=(4.4, 4.4))
    ax = plt.subplot(111, polar=True)
    ax.plot(ang, vals, linewidth=2, color="#d9534f")
    ax.fill(ang, vals, alpha=0.30, color="#d9534f")
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_yticklabels([])
    fig.patch.set_facecolor("white")
    return fig


def show_result(text, r):
    st.markdown(f"### {r['bucket']}" + (f"  ·  {r['type']}" if r["type"] != "-" else ""))
    a, b, c = st.columns(3)
    a.metric("ML spam probability", f"{r['p']*100:.1f}%")
    b.metric("Final risk (ML + rules + sender)", f"{r['risk']*100:.1f}%")
    c.metric("Red flags found", len(r["flags"]))
    st.progress(min(r["risk"], 1.0))
    left, right = st.columns([3, 2])
    with left:
        st.markdown("**🧠 Why? (red = pushes toward spam, green = toward safe)**")
        st.markdown(highlight(text, r["all"]), unsafe_allow_html=True)
        st.caption("Top signals: " + signals_text(r["signals"]))
        if r["flags"]:
            st.markdown("**🚩 Phishing / red flags**")
            for n, d, w in r["flags"]:
                st.write(f"- {n}" + (f": `{d}`" if d else ""))
        s_score, s_why = r["sender"]
        if s_score is not None:
            st.markdown(f"**📮 Sender reputation:** risk {s_score*100:.0f}%")
            for n, d in s_why:
                st.write(f"- {n}: `{d}`")
    with right:
        st.markdown("**🧬 Email DNA**")
        st.pyplot(radar(r["dna"]))


def prompts(text):
    zero = ("You are an email security assistant. Classify the email below as SPAM, REVIEW or SAFE and give a one-line reason.\n\n"
            f'Email:\n"""{text}"""\n\nAnswer in the format: Label - Reason')
    few = ("Classify each email as SPAM or SAFE.\n\n"
           'Email: "You won Rs 50,000! Click bit.ly/claim and send your bank details." -> SPAM\n'
           'Email: "Tomorrow\'s DBMS lab is moved to 2 PM in Lab 3." -> SAFE\n'
           'Email: "Your account is blocked. Verify your password within 24 hours." -> SPAM\n'
           f'Email: "{text}" ->')
    return zero, few


def add_feedback(text, label):
    fb = Path("data/feedback.csv")
    pd.DataFrame({"text": [text], "label": [label]}).to_csv(fb, mode="a", header=not fb.exists(), index=False)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🛡️ MailShield")
    st.caption("Explainable email spam triage")
    model_name = st.radio("Active model", list(MODELS))
    spam_th = st.slider("Spam at risk ≥", 0.5, 0.95, 0.70, 0.05)
    review_th = st.slider("Review at risk ≥", 0.1, spam_th - 0.05, min(0.40, spam_th - 0.05), 0.05)
    st.divider()
    source = st.radio("Scan source", ["Built-in sample emails", "Upload CSV"])
    up = st.file_uploader("CSV: a text column (optional sender, label)", type="csv") if source == "Upload CSV" else None
    speed = st.slider("Seconds per email", 0.0, 2.0, 0.5, 0.1)
    key = (source, up.name if up else None)
    if ss.src != key:
        if source == "Built-in sample emails":
            ss.queue = read_queue(pd.read_csv("data/sample_scan.csv"))
        elif up is not None:
            ss.queue = read_queue(pd.read_csv(up, encoding="latin-1"))
        else:
            ss.queue = []
        ss.update(src=key, idx=0, results=[], running=False)
    c1, c2, c3 = st.columns(3)
    if c1.button("▶️", help="Start / resume"):
        ss.running = True
    if c2.button("⏸️", help="Pause"):
        ss.running = False
    if c3.button("🔄", help="Reset"):
        ss.update(idx=0, results=[], running=False)
    st.caption(f"{ss.idx} / {len(ss.queue)} scanned")

# ------------------------------------------------------------------ one scan step
if ss.running and ss.idx < len(ss.queue):
    item = ss.queue[ss.idx]
    r = analyze(item["text"], item["sender"], model_name, spam_th, review_th)
    ss.results.append({"#": ss.idx + 1, "Sender": item["sender"], "Email": item["text"][:80].replace("\n", " "),
                       "Spam %": round(r["p"] * 100, 1), "Risk %": round(r["risk"] * 100, 1), "Bucket": r["bucket"],
                       "Type": r["type"], "Top signals": signals_text(r["signals"]),
                       "_y": item["y"], "_pred": int(r["risk"] >= 0.5)})
    ss.idx += 1

res = pd.DataFrame(ss.results)
tabs = st.tabs(["📬 Scanner", "🔍 Check an email", "🧪 Risk simulator", "📊 Model performance", "📈 Analytics",
                "🧠 Teach the model"])

# ------------------------------------------------------------------ 1. scanner
with tabs[0]:
    cnt = res["Bucket"].value_counts() if not res.empty else pd.Series(dtype=int)
    a, b, c, d = st.columns(4)
    a.metric("Scanned", len(res))
    b.metric(SPAM_B, int(cnt.get(SPAM_B, 0)))
    c.metric(REVIEW_B, int(cnt.get(REVIEW_B, 0)))
    d.metric(SAFE_B, int(cnt.get(SAFE_B, 0)))
    if ss.queue:
        st.progress(min(ss.idx / len(ss.queue), 1.0))
    if res.empty:
        st.info("Press ▶️ in the sidebar to start scanning. Use 'Upload CSV' in the sidebar to scan your own emails.")
    else:
        st.dataframe(res.drop(columns=["_y", "_pred"]).iloc[::-1], hide_index=True)
        lab = res.dropna(subset=["_y"])
        if len(lab):
            st.subheader("Confusion matrix of this batch (spam = risk ≥ 50%)")
            tp, fn, fp, tn = counts(lab["_y"].astype(int), lab["_pred"])
            m1, m2 = st.columns([1, 1])
            m1.markdown(cm_html(tp, fn, fp, tn), unsafe_allow_html=True)
            m2.dataframe(pd.DataFrame(rates(tp, fn, fp, tn), index=["value"]).T.round(3))
        st.line_chart(res.set_index("#")["Risk %"])

# ------------------------------------------------------------------ 2. check an email
with tabs[1]:
    st.subheader("Paste any email")
    examples = {
        "— choose an example —": ("", ""),
        "Prize scam": ("Subject: CONGRATULATIONS!!! You WON Rs 50,000. Click http://bit.ly/claim-now and send your bank account details within 24 hours!", "winner@free-gifts.ru"),
        "Fake internship fee": ("Subject: Internship offer from TCS. You are selected! Pay a registration fee of Rs 2500 immediately to confirm your seat.", "hr@tcs-careers-india.xyz"),
        "Normal class mail": ("Subject: Lab moved. Hi all, tomorrow's DBMS lab is shifted to 2 PM in Lab 3. Please bring your record books.", "prof.rao@bmsit.ac.in"),
        "Real OTP message": ("Subject: ERP OTP. Your one time password is 482913 and is valid for 10 minutes. Do not share this code with anyone.", "noreply@bmsit.ac.in"),
    }

    def load_example():
        ss.paste, ss.sender = examples[ss.example]

    st.selectbox("Try an example", list(examples), key="example", on_change=load_example)
    st.text_area("Email text (subject + body)", height=170, key="paste", placeholder="Paste an email here...")
    st.text_input("Sender address (optional, enables sender reputation)", key="sender", placeholder="name@domain.com")
    if st.button("Run triage check", type="primary") and ss.paste.strip():
        ss.last = (ss.paste, ss.sender, analyze(ss.paste, ss.sender, model_name, spam_th, review_th))
    if ss.last:
        text, sender, r = ss.last
        show_result(text, r)
        st.markdown("**Was this wrong? Teach the model:**")
        f1, f2 = st.columns(2)
        if f1.button("Mark as SPAM"):
            add_feedback(text, "spam")
            st.success("Saved. Retrain in the 'Teach the model' tab.")
        if f2.button("Mark as SAFE"):
            add_feedback(text, "ham")
            st.success("Saved. Retrain in the 'Teach the model' tab.")
        with st.expander("Module 3 - the same task as a prompt (zero-shot vs few-shot)"):
            z, f = prompts(text)
            st.markdown("**Zero-shot prompt** (instruction only):")
            st.code(z, language=None)
            st.markdown("**Few-shot prompt** (instruction + labelled examples):")
            st.code(f, language=None)
            st.caption("You could paste these into any LLM. Our trained model needs no prompt: it learned from thousands of labelled emails.")

# ------------------------------------------------------------------ 3. risk simulator
with tabs[2]:
    st.subheader("How does wording change the score?")
    if not ss.last:
        st.info("First run a check in the 'Check an email' tab, then come back here.")
    else:
        text0, sender0, r0 = ss.last
        if ss.sim_src != text0:
            ss.sim_text, ss.sim_src = text0, text0
        st.text_area("Edit the email freely", height=140, key="sim_text")
        o1, o2, o3 = st.columns(3)
        rm_links = o1.checkbox("Remove links")
        rm_urg = o1.checkbox("Remove urgency words")
        rm_money = o2.checkbox("Remove money / prize words")
        rm_sens = o2.checkbox("Remove password / OTP requests")
        lower = o3.checkbox("Lowercase everything")
        no_excl = o3.checkbox("Remove '!' marks")
        t = ss.sim_text
        if rm_links:
            t = strip_links(t)
        if rm_urg:
            t = strip_words(t, URGENT)
        if rm_money:
            t = strip_words(t, MONEY)
        if rm_sens:
            t = strip_words(t, SENSITIVE)
        if lower:
            t = t.lower()
        if no_excl:
            t = t.replace("!", "")
        r1 = analyze(t, sender0, model_name, spam_th, review_th)
        x, y, z = st.columns(3)
        x.metric("Original risk", f"{r0['risk']*100:.1f}%")
        y.metric("New risk", f"{r1['risk']*100:.1f}%", f"{(r1['risk']-r0['risk'])*100:+.1f} pts", delta_color="inverse")
        z.metric("New bucket", r1["bucket"])
        st.markdown("**Modified email:**")
        st.markdown(highlight(t, r1["all"]), unsafe_allow_html=True)
        st.bar_chart(pd.DataFrame({"Original": r0["dna"], "Modified": r1["dna"]}))
        st.caption("Each bar is one 'DNA' trait. Watch which traits drop as you remove spam-like wording.")

# ------------------------------------------------------------------ 4. model performance
with tabs[3]:
    st.subheader("Confusion matrix and metrics")
    if evald is None:
        st.warning("Run train.py to generate evaluation data.")
    else:
        th = st.slider("Decision threshold (spam if probability ≥)", 0.1, 0.9, 0.5, 0.05)
        yt = evald["y_test"]
        cols = st.columns(len(MODELS))
        rows = {}
        for col, (n, p) in zip(cols, evald["proba"].items()):
            tp, fn, fp, tn = counts(yt, (p >= th).astype(int))
            col.markdown(f"**{n}**")
            col.markdown(cm_html(tp, fn, fp, tn), unsafe_allow_html=True)
            rows[n] = rates(tp, fn, fp, tn)
        st.dataframe(pd.DataFrame(rows).round(4))
        st.caption("TP = spam caught · TN = safe mail left alone · FP = safe mail wrongly flagged (worst for users) · FN = spam missed.")
        with st.expander("Formulas from class"):
            st.markdown("- Accuracy = (TP+TN)/(TP+TN+FP+FN)\n- Precision = TP/(TP+FP)\n- Recall = TP/(TP+FN)\n"
                        "- Specificity = TN/(TN+FP)\n- F1 = 2·P·R/(P+R)\n- False-positive rate = FP/(FP+TN)")
        if evald.get("campus_y") is not None:
            st.subheader("Out-of-domain test: 60 hand-written campus emails (never used in training)")
            cols = st.columns(len(MODELS))
            rows = {}
            for col, (n, p) in zip(cols, evald["campus_proba"].items()):
                tp, fn, fp, tn = counts(evald["campus_y"], (p >= th).astype(int))
                col.markdown(f"**{n}**")
                col.markdown(cm_html(tp, fn, fp, tn), unsafe_allow_html=True)
                rows[n] = rates(tp, fn, fp, tn)
            st.dataframe(pd.DataFrame(rows).round(4))
    if meta:
        st.subheader("Is the model over- or under-fitting?")
        rows = [{"Model": n, "Train accuracy": m["train_accuracy"], "CV F1": m["cv_f1"], "Test accuracy": m["test"]["accuracy"],
                 "Gap (train - test)": m["train_accuracy"] - m["test"]["accuracy"],
                 **({"Campus-test accuracy": m["campus_test"]["accuracy"]} if "campus_test" in m else {})}
                for n, m in meta["models"].items()]
        st.dataframe(pd.DataFrame(rows).set_index("Model").round(4))
        st.caption("A big train-vs-test gap means overfitting. A low score everywhere means underfitting.")
        st.subheader("Hyperparameter tuning (grid search with cross-validation)")
        st.dataframe(pd.DataFrame(meta["tuning"]), hide_index=True)
        st.caption(f"Chosen features: {meta['features']} · {meta['n_emails']:,} emails ({meta['n_spam']:,} spam) · sources: {meta['sources']}")
    if Path("models/errors.csv").exists():
        st.subheader("Error analysis: emails the Logistic Regression got wrong")
        st.dataframe(pd.read_csv("models/errors.csv"), hide_index=True)

# ------------------------------------------------------------------ 5. analytics
with tabs[4]:
    st.subheader("Spam analytics")
    a1, a2 = st.columns(2)
    with a1:
        st.markdown("**This scan session**")
        if res.empty:
            st.info("Run the scanner to see session statistics.")
        else:
            flagged = res[res["Type"] != "-"]
            st.write(f"{len(flagged)} of {len(res)} emails flagged ({len(flagged)/len(res)*100:.0f}%).")
            if len(flagged):
                st.bar_chart(flagged["Type"].value_counts())
            hist, edges = np.histogram(res["Risk %"], bins=10, range=(0, 100))
            st.bar_chart(pd.Series(hist, index=[f"{int(e)}-{int(e)+10}" for e in edges[:-1]], name="emails"))
            st.caption("Risk distribution: a good model shows two clear clusters (safe on the left, spam on the right).")
    with a2:
        st.markdown(f"**Strongest words learned by {model_name}**")
        w = weights(model_name)
        order = np.argsort(w)
        top_spam = pd.Series(w[order[::-1][:12]], index=FEATS[order[::-1][:12]])
        top_ham = pd.Series(-w[order[:12]], index=FEATS[order[:12]])
        st.caption("Spam words"), st.bar_chart(top_spam)
        st.caption("Safe words"), st.bar_chart(top_ham)
    if meta:
        st.markdown(f"**Training data:** {meta['n_emails']:,} emails, {meta['n_spam']/meta['n_emails']*100:.0f}% spam. "
                    f"By source: {meta['sources']}")

# ------------------------------------------------------------------ 6. teach the model
with tabs[5]:
    st.subheader("Adaptive learning (human-in-the-loop)")
    st.write("Correct a wrong decision, then retrain. The corrections are added to the training data.")
    if not res.empty:
        pick = st.selectbox("Correct a scanned email", [f"{r['#']}: {r['Email'][:60]}" for _, r in res.iterrows()])
        i = int(pick.split(":")[0]) - 1
        k1, k2 = st.columns(2)
        if k1.button("This is SPAM", key="fb_s"):
            add_feedback(ss.queue[i]["text"], "spam")
            st.success("Saved.")
        if k2.button("This is SAFE", key="fb_h"):
            add_feedback(ss.queue[i]["text"], "ham")
            st.success("Saved.")
    fb = Path("data/feedback.csv")
    n_fb = len(pd.read_csv(fb)) if fb.exists() else 0
    st.metric("Corrections saved", n_fb)
    if n_fb:
        st.dataframe(pd.read_csv(fb).tail(10), hide_index=True)
    can_retrain = Path("data/enron.csv").exists() or Path("data/emails.csv").exists()
    if not can_retrain:
        st.info("Retraining is available in the local version of MailShield (it needs the full training dataset).")
    elif st.button("🔁 Retrain now (quick mode)"):
        with st.spinner("Retraining ..."):
            out = subprocess.run([sys.executable, "train.py", "--fast"], capture_output=True, text=True)
        st.code((out.stdout or out.stderr)[-1500:], language=None)
        st.cache_resource.clear()
        st.rerun()
    prev = Path("models/metrics_prev.json")
    if prev.exists() and meta:
        old = json.loads(prev.read_text())
        st.markdown("**Before vs after the last retrain**")
        rows = []
        for n in meta["models"]:
            o, nw = old["models"].get(n), meta["models"][n]
            if o:
                rows.append({"Model": n, "Test F1 before": o["test"]["f1"], "Test F1 after": nw["test"]["f1"],
                             "Campus acc before": o.get("campus_test", {}).get("accuracy"),
                             "Campus acc after": nw.get("campus_test", {}).get("accuracy")})
        st.dataframe(pd.DataFrame(rows).set_index("Model").round(4))

# ------------------------------------------------------------------ auto-scan loop
if ss.running:
    if ss.idx < len(ss.queue):
        time.sleep(speed)
    else:
        ss.running = False
    st.rerun()
