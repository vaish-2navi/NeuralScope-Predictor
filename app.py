import streamlit as st
import pickle
import numpy as np
from pathlib import Path

# ---------------- Page setup ----------------
st.set_page_config(
    page_title="NeuralScope | ML Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).parent / "model.pkl"

# ---------------- Dark modern UI ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root { color-scheme: dark; }
.stApp {
    background:
      radial-gradient(circle at 12% 0%, rgba(87, 70, 210, .16), transparent 28%),
      radial-gradient(circle at 95% 12%, rgba(0, 204, 180, .10), transparent 25%),
      #090d18;
    color: #edf2ff;
    font-family: 'Inter', sans-serif;
}
[data-testid="stHeader"] { background: rgba(9,13,24,.72); }
[data-testid="stSidebar"] {
    background: #0d1322;
    border-right: 1px solid #202b40;
}
.block-container { padding-top: 2.1rem; padding-bottom: 3rem; max-width: 1450px; }
h1, h2, h3, p, label { color: #edf2ff; }
.hero {
    padding: 30px 32px;
    border: 1px solid #293653;
    border-radius: 22px;
    background: linear-gradient(120deg, rgba(23,34,61,.96), rgba(17,29,46,.92));
    box-shadow: 0 18px 60px rgba(0,0,0,.20);
    margin-bottom: 25px;
}
.eyebrow {
    display: inline-block;
    color: #67e8d5;
    background: rgba(49,212,190,.10);
    border: 1px solid rgba(49,212,190,.22);
    border-radius: 999px;
    padding: 6px 11px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.5px;
}
.hero h1 { font-size: 36px; font-weight: 800; margin: 15px 0 8px 0; letter-spacing: -1.2px; }
.hero p { color: #aab8d2; font-size: 15px; margin: 0; }
.panel {
    background: rgba(17,24,39,.86);
    border: 1px solid #263149;
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 18px;
}
.panel-title { font-size: 17px; font-weight: 700; margin-bottom: 5px; }
.panel-subtitle { color: #8e9bb5; font-size: 12px; margin-bottom: 18px; }
div[data-testid="stNumberInput"] input {
    background: #0b1220 !important;
    color: #f0f5ff !important;
    border: 1px solid #2b3852 !important;
    border-radius: 10px !important;
}
div[data-testid="stNumberInput"] button {
    background: #172238 !important;
    color: #b9c8e3 !important;
    border-color: #2b3852 !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #6d5dfc, #3987f6);
    color: white;
    border: 0;
    border-radius: 11px;
    min-height: 48px;
    font-weight: 700;
    box-shadow: 0 8px 24px rgba(75,105,245,.25);
}
.stButton > button {
    border-radius: 10px;
    border: 1px solid #2b3852;
}
.metric-card {
    background: linear-gradient(135deg, #171f34, #111a2a);
    border: 1px solid #2b3852;
    border-radius: 16px;
    padding: 20px;
}
.metric-label { color: #8f9db8; font-size: 12px; margin-bottom: 8px; }
.metric-value { color: #73ead8; font-size: 28px; font-weight: 800; }
.note {
    border-left: 3px solid #6d5dfc;
    padding: 10px 13px;
    background: rgba(109,93,252,.08);
    color: #aab8d2;
    font-size: 12px;
    border-radius: 0 8px 8px 0;
}
div[data-testid="stAlert"] { border-radius: 12px; }
hr { border-color: #263149; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

# ---------------- Header ----------------
st.markdown("""
<div class="hero">
  <span class="eyebrow">MACHINE LEARNING • LIVE INFERENCE</span>
  <h1>NeuralScope <span style="color:#73ead8">/</span> Predictor</h1>
  <p>A focused workspace for model inference, class probabilities, and feature inspection.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- Model load ----------------
try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file nahi mili. `app.py` aur `model.pkl` ko same folder mein rakho.")
    st.stop()
except Exception as e:
    st.error(f"Model load nahi hua: {e}")
    st.info("Pickle model ko wahi Python/scikit-learn environment chahiye jisme compatible version available ho.")
    st.stop()

n_features = int(getattr(model, "n_features_in_", 11))
if n_features != 11:
    st.warning(f"Uploaded model {n_features} features expect karta hai, jabki yeh UI PC1–PC11 ke liye set hai.")
    st.stop()

classes = list(map(str, getattr(model, "classes_", ["B", "M"])))

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("## 🧬 NeuralScope")
    st.caption("MODEL WORKSPACE")
    st.divider()
    st.markdown("**Model details**")
    st.markdown("Algorithm  \n`Random Forest Classifier`")
    st.markdown("Input dimensions  \n`11 PCA components`")
    st.markdown("Output classes")
    st.code(" • ".join(classes), language=None)
    st.divider()
    st.markdown("**How to use**")
    st.markdown("1. Enter PC1–PC11 values.\n2. Click **Run prediction**.\n3. Review the output and probabilities.")
    st.markdown('<div class="note">If training used a scaler or PCA transformer, use the same fitted preprocessing pipeline before prediction.</div>', unsafe_allow_html=True)
    if st.button("↻ Reset inputs", use_container_width=True):
        for i in range(1, 12):
            st.session_state[f"pc_{i}"] = 0.0
        st.session_state.pop("prediction_done", None)
        st.rerun()

# ---------------- Main content ----------------
left, right = st.columns([1.55, 1], gap="large")

with left:
    st.markdown('<div class="panel"><div class="panel-title">Input features</div><div class="panel-subtitle">Provide the 11 PCA component scores expected by your trained model.</div>', unsafe_allow_html=True)
    values = []
    for row_start in (1, 4, 7, 10):
        cols = st.columns(3)
        for offset, col in enumerate(cols):
            i = row_start + offset
            if i > 11:
                continue
            with col:
                val = st.number_input(
                    f"PC{i}",
                    min_value=-1_000_000.0,
                    max_value=1_000_000.0,
                    value=0.0,
                    step=0.1,
                    format="%.4f",
                    key=f"pc_{i}",
                )
                values.append((i, val))
    values = [v for _, v in sorted(values)]
    st.markdown('</div>', unsafe_allow_html=True)

    run_prediction = st.button("✦  Run prediction", type="primary", use_container_width=True)

with right:
    st.markdown('<div class="panel"><div class="panel-title">Inference overview</div><div class="panel-subtitle">Your current model configuration</div>', unsafe_allow_html=True)
    st.markdown(f"""
      <div class="metric-card">
        <div class="metric-label">MODEL TYPE</div>
        <div style="font-size:18px;font-weight:700;">Random Forest</div>
      </div><br>
      <div class="metric-card">
        <div class="metric-label">INPUT FEATURES</div>
        <div class="metric-value">11 <span style="font-size:13px;color:#9eabc3;font-weight:500;">PCA components</span></div>
      </div><br>
      <div class="metric-card">
        <div class="metric-label">PREDICTION CLASSES</div>
        <div style="font-size:18px;font-weight:700;">{' / '.join(classes)}</div>
      </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if run_prediction:
    try:
        X = np.array([values], dtype=float)
        pred = model.predict(X)[0]
        st.divider()
        st.markdown("## Prediction result")
        result_col, probability_col = st.columns([1, 1.5], gap="large")
        with result_col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">PREDICTED CLASS</div>
              <div class="metric-value">{str(pred)}</div>
              <div style="color:#8f9db8;font-size:12px;margin-top:8px;">Output returned by the trained model</div>
            </div>
            """, unsafe_allow_html=True)
        with probability_col:
            st.markdown('<div class="panel-title">Class probabilities</div>', unsafe_allow_html=True)
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[0]
                st.bar_chart(
                    {str(label): float(prob) for label, prob in zip(model.classes_, probs)},
                    color="#73ead8",
                )
                st.caption("Probabilities are model estimates, not a guarantee of correctness.")
            else:
                st.info("This model does not expose predict_proba().")
        with st.expander("View submitted feature values"):
            st.dataframe(
                {"Feature": [f"PC{i}" for i in range(1, 12)], "Value": values},
                use_container_width=True,
                hide_index=True,
            )
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("<hr>", unsafe_allow_html=True)
st.caption("NeuralScope • Local ML dashboard • Interpret class labels using the original dataset documentation.")

