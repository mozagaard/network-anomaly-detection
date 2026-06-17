import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ======================
# PAGE CONFIG
# ======================

st.set_page_config(
    page_title="Network Anomaly Detection",
    page_icon="🛡️",
    layout="wide"
)

# ======================
# CUSTOM CSS
# ======================

st.markdown("""
<style>

.main {
    background-color: #0f172a;
}

[data-testid="stMetric"]{
    background: #111827;
    border: 1px solid #374151;
    padding: 15px;
    border-radius: 15px;
    text-align:center;
}

.stButton > button {
    width:100%;
    background:#2563eb;
    color:white;
    border:none;
    border-radius:10px;
    height:50px;
    font-size:18px;
    font-weight:bold;
}

.stButton > button:hover{
    background:#1d4ed8;
}

.block-container{
    padding-top:2rem;
}

</style>
""", unsafe_allow_html=True)

# ======================
# LOAD MODEL
# ======================

xgb_binary = joblib.load("model/xgb_binary.pkl")
xgb_attack = joblib.load("model/xgb_attack.pkl")

attack_label_names = joblib.load(
    "model/attack_label_names.pkl"
)

feature_names = joblib.load(
    "model/feature_names.pkl"
)

# ======================
# HERO SECTION
# ======================

st.markdown("""
<div style="
padding:25px;
border-radius:20px;
background:linear-gradient(135deg,#0f172a,#1e293b);
color:white;
margin-bottom:20px;
">

<h1>🛡️ Network Anomaly Detection System</h1>

<p>
Real-Time Intrusion Detection using Cascading XGBoost Classifier
</p>

</div>
""", unsafe_allow_html=True)

# ======================
# KPI SECTION
# ======================

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric("📊 Dataset", "KDD Cup 99")

with k2:
    st.metric("🤖 Stage 1", "Binary XGBoost")

with k3:
    st.metric("⚔️ Stage 2", "Attack Classifier")

with k4:
    st.metric("🎯 Accuracy", "99%+")

st.divider()

# ======================
# INPUT SECTION
# ======================

st.subheader("📡 Network Traffic Parameters")

left, right = st.columns(2)

with left:
    duration = st.number_input(
        "Duration",
        min_value=0,
        value=0
    )

    src_bytes = st.number_input(
        "Source Bytes",
        min_value=0,
        value=181
    )

    dst_bytes = st.number_input(
        "Destination Bytes",
        min_value=0,
        value=5450
    )

    count = st.number_input(
        "Count",
        min_value=0,
        value=2
    )

    wrong_fragment = st.number_input(
        "Wrong Fragment",
        min_value=0,
        value=0
    )

with right:

    protocol = st.selectbox(
        "Protocol Type",
        [0, 1, 2],
        format_func=lambda x:
        ["icmp", "tcp", "udp"][x]
    )

    srv_count = st.number_input(
        "Srv Count",
        min_value=0,
        value=2
    )

    same_srv_rate = st.slider(
        "Same Service Rate",
        0.0,
        1.0,
        0.5
    )

    diff_srv_rate = st.slider(
        "Different Service Rate",
        0.0,
        1.0,
        0.5
    )

    serror_rate = st.slider(
        "Serror Rate",
        0.0,
        1.0,
        0.0
    )

analyze = st.button(
    "🚀 Analyze Traffic"
)

# ======================
# PREDICTION
# ======================

if analyze:

    input_dict = {
        col: 0 for col in feature_names
    }

    input_dict["duration"] = duration
    input_dict["protocol_type"] = protocol
    input_dict["src_bytes"] = src_bytes
    input_dict["dst_bytes"] = dst_bytes
    input_dict["wrong_fragment"] = wrong_fragment
    input_dict["same_srv_rate"] = same_srv_rate
    input_dict["diff_srv_rate"] = diff_srv_rate
    input_dict["serror_rate"] = serror_rate
    input_dict["count"] = count
    input_dict["srv_count"] = srv_count

    input_df = pd.DataFrame([input_dict])

    binary_pred = xgb_binary.predict(
        input_df
    )[0]

    binary_proba = xgb_binary.predict_proba(
        input_df
    )[0]

    st.divider()

    st.subheader("🔎 Detection Result")

    # ======================
    # NORMAL
    # ======================

    if binary_pred == 0:

        st.success(
            f"✅ NORMAL TRAFFIC | Confidence: {binary_proba[0]*100:.2f}%"
        )

        fig = px.bar(
            x=["Normal", "Anomaly"],
            y=[
                binary_proba[0],
                binary_proba[1]
            ],
            color=[
                "Normal",
                "Anomaly"
            ],
            color_discrete_sequence=[
                "#22c55e",
                "#ef4444"
            ]
        )

        fig.update_layout(
            title="Prediction Confidence",
            template="plotly_dark",
            height=350,
            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ======================
    # ATTACK
    # ======================

    else:

        st.markdown(f"""
        <div style="
        background:#7f1d1d;
        padding:20px;
        border-radius:15px;
        color:white;
        text-align:center;
        ">
        <h2>🚨 ATTACK DETECTED</h2>
        <h3>Confidence: {binary_proba[1]*100:.2f}%</h3>
        </div>
        """, unsafe_allow_html=True)

        attack_pred = xgb_attack.predict(
            input_df
        )[0]

        attack_proba = xgb_attack.predict_proba(
            input_df
        )[0]

        attack_name = attack_label_names.get(
            attack_pred,
            "Unknown"
        )

        st.warning(
            f"⚔️ Attack Type: {attack_name.upper()}"
        )

        labels = [
            attack_label_names[i]
            for i in range(len(attack_proba))
        ]

        attack_df = pd.DataFrame({
            "Attack Type": labels,
            "Probability": attack_proba
        })

        fig_attack = px.bar(
            attack_df,
            x="Attack Type",
            y="Probability",
            color="Probability",
            color_continuous_scale="Reds"
        )

        fig_attack.update_layout(
            template="plotly_dark",
            height=450,
            title="Attack Classification Confidence"
        )

        st.plotly_chart(
            fig_attack,
            use_container_width=True
        )

    # ======================
    # FEATURE IMPORTANCE
    # ======================

    st.subheader(
        "📈 Feature Importance"
    )

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance":
        xgb_binary.feature_importances_
    })

    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
        .head(10)
    )

    fig_imp = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Blues"
    )

    fig_imp.update_layout(
        template="plotly_dark",
        height=500
    )

    st.plotly_chart(
        fig_imp,
        use_container_width=True
    )

# ======================
# INFO SECTION
# ======================

else:

    st.info(
        "👆 Configure the traffic parameters and click Analyze Traffic."
    )

    st.subheader(
        "How Cascading Classification Works"
    )

    st.markdown("""
### Stage 1 — Binary Detection
Determine whether incoming traffic is Normal or Anomalous.

### Stage 2 — Attack Classification
If an anomaly is detected, the second model identifies the attack category.

### Benefits
- Faster detection process
- Reduced false positives
- Better attack categorization
- Interpretable machine learning workflow
""")