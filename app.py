"""Early Academic Intervention Dashboard - Machine Problem 3 (Streamlit)."""
import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from features import FEATURES, engineer_features

ART_DIR = "artifacts"
DATA_PATH = "data/augmented_balanced_dataset_m.csv"

st.set_page_config(page_title="LMS Student Performance Predictor", page_icon="🎓", layout="wide")


# ----------------------------------------------------------------- artifacts
@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    model_path = os.path.join(ART_DIR, "model.joblib")
    if not os.path.exists(model_path):
        if not os.path.exists(DATA_PATH):
            return None
        from train_model import train  # first-run fallback: train from the CSV
        train(DATA_PATH, ART_DIR)
    bundle = joblib.load(model_path)
    with open(os.path.join(ART_DIR, "metrics.json")) as f:
        metrics = json.load(f)
    with open(os.path.join(ART_DIR, "stats.json")) as f:
        stats = json.load(f)
    return bundle, metrics, stats


art = load_artifacts()
if art is None:
    st.error("Model artifacts not found. Place `augmented_balanced_dataset_m.csv` in `data/` "
             "and run `python train_model.py`, then reload.")
    st.stop()
bundle, metrics, stats = art
model, scaler, classes = bundle["model"], bundle["scaler"], bundle["classes"]
feature_columns = bundle["feature_columns"]
LOW_IDX = classes.index("Low")
HIGH_IDX = classes.index("High")


def predict(values: dict):
    row = pd.DataFrame([values], columns=FEATURES)
    X = engineer_features(row).reindex(columns=feature_columns, fill_value=0)
    proba = model.predict_proba(scaler.transform(X))[0]
    return float(proba[HIGH_IDX]), float(proba[LOW_IDX])


def percentile_of(feature: str, value: float) -> float:
    s = stats["features"][feature]
    return float(np.interp(value, s["quantiles"], stats["percentile_grid"]))


# ------------------------------------------------------------ input defaults
def feat_max(f):
    return int(max(stats["features"][f]["max"], 1))


def set_values(source: dict):
    for f in FEATURES:
        v = int(round(source[f])) if source else 0
        st.session_state[f] = min(max(v, 0), feat_max(f))


if "initialized" not in st.session_state:
    set_values(stats["class_medians"]["High"])
    st.session_state["initialized"] = True

# ------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Student LMS Activity")
    c1, c2, c3 = st.columns(3)
    c1.button("High", help="Typical High-rated student (class medians)", use_container_width=True,
              on_click=set_values, args=(stats["class_medians"]["High"],))
    c2.button("At-Risk", help="Typical Low-rated student (class medians)", use_container_width=True,
              on_click=set_values, args=(stats["class_medians"]["Low"],))
    c3.button("Reset", help="Set everything to zero", use_container_width=True,
              on_click=set_values, args=(None,))
    st.caption("Presets use the median activity of each rating class in the dataset.")
    inputs = {}
    for f in FEATURES:
        inputs[f] = st.slider(f, 0, feat_max(f), key=f)

# ---------------------------------------------------------------------- main
st.title("🎓 Early Academic Intervention Dashboard")
st.write("Enter a student's midterm LMS activity to predict whether they fall in the "
         "**High** or **Low** performance rating category.")

tab_pred, tab_insight, tab_model = st.tabs(["Prediction", "Feature Insights", "Model Performance"])

p_high, p_low = predict(inputs)
label = "Low" if p_low >= p_high else "High"
conf = max(p_low, p_high)

with tab_pred:
    left, right = st.columns([1, 1])
    with left:
        if label == "Low":
            st.error(f"### Predicted rating: LOW  \nConfidence: **{conf:.1%}**")
        else:
            st.success(f"### Predicted rating: HIGH  \nConfidence: **{conf:.1%}**")

        if label == "Low" and conf >= 0.85:
            note = ("**High risk.** Recommend prompt outreach: contact the student, review missed "
                    "quizzes/submissions, and point to tutoring or support resources.")
        elif label == "Low":
            note = ("**Moderate risk.** The model leans Low but is less certain. Monitor "
                    "engagement over the coming weeks and send a friendly check-in.")
        elif conf >= 0.85:
            note = "**Low risk.** Engagement looks healthy; no intervention needed."
        else:
            note = ("**Borderline.** The model leans High but is less certain. Keep an eye on "
                    "activity trends.")
        st.info(note)
    with right:
        fig = go.Figure(go.Bar(
            x=[p_high * 100, p_low * 100], y=["High", "Low"], orientation="h",
            marker_color=["#2e8b57", "#c0392b"],
            text=[f"{p_high:.1%}", f"{p_low:.1%}"], textposition="auto"))
        fig.update_layout(title="Class probabilities", xaxis_title="Probability (%)",
                          xaxis_range=[0, 100], height=280, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

with tab_insight:
    rows = []
    for f in FEATURES:
        pct = percentile_of(f, inputs[f])
        level = "Low" if pct < 25 else ("High" if pct > 75 else "Moderate")
        rows.append({"Feature": f, "Value": inputs[f], "Dataset median": stats["features"][f]["median"],
                     "Percentile": round(pct, 1), "Level": level})
    df_ins = pd.DataFrame(rows)
    colors = df_ins["Level"].map({"Low": "#c0392b", "Moderate": "#e0a800", "High": "#2e8b57"})
    fig = go.Figure(go.Bar(x=df_ins["Percentile"], y=df_ins["Feature"], orientation="h",
                           marker_color=colors, text=df_ins["Percentile"].round(0).astype(int).astype(str) + "th",
                           textposition="auto"))
    fig.update_layout(title="Where this student sits relative to the dataset", xaxis_title="Percentile",
                      xaxis_range=[0, 100], height=420, yaxis_autorange="reversed",
                      margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df_ins, hide_index=True, use_container_width=True)
    weak = df_ins.loc[df_ins["Level"] == "Low", "Feature"].tolist()
    strong = df_ins.loc[df_ins["Level"] == "High", "Feature"].tolist()
    st.write("**Below the 25th percentile:** " + (", ".join(weak) if weak else "none"))
    st.write("**Above the 75th percentile:** " + (", ".join(strong) if strong else "none"))
    st.caption("Levels compare the entered value with the training dataset's distribution "
               "(Low < 25th percentile, High > 75th). They describe the inputs, not model weights.")

with tab_model:
    m1, m2, m3 = st.columns(3)
    m1.metric("Test accuracy", f"{metrics['accuracy']:.2%}")
    m2.metric("Training / test rows", f"{metrics['n_train']} / {metrics['n_test']}")
    m3.metric("Engineered features", len(feature_columns))

    cm = np.array(metrics["confusion_matrix"])
    fig = go.Figure(go.Heatmap(z=cm, x=[f"Pred {c}" for c in classes], y=[f"Actual {c}" for c in classes],
                               text=cm, texttemplate="%{text}", colorscale="Blues", showscale=False))
    fig.update_layout(title="Confusion matrix (held-out test set)", height=380,
                      yaxis_autorange="reversed", margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

    rep = metrics["report"]
    st.dataframe(pd.DataFrame({c: {"Precision": rep[c]["precision"], "Recall": rep[c]["recall"],
                                   "F1-score": rep[c]["f1-score"], "Support": rep[c]["support"]}
                               for c in classes}).T.style.format(
        {"Precision": "{:.2%}", "Recall": "{:.2%}", "F1-score": "{:.2%}", "Support": "{:.0f}"}),
        use_container_width=True)
    st.caption("Model: soft VotingClassifier (HistGradientBoosting + ExtraTrees), 80/20 stratified split, "
               "random_state=42, log1p + pairwise sum/product features, StandardScaler fit on training data only.")
