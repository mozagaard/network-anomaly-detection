import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# load semua model
xgb_binary = joblib.load('model/xgb_binary.pkl')
xgb_attack = joblib.load('model/xgb_attack.pkl')
attack_label_names = joblib.load('model/attack_label_names.pkl')
feature_names = joblib.load('model/feature_names.pkl')

st.set_page_config(
    page_title="Network Anomaly Detection",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Network Anomaly Detection")
st.markdown("Deteksi serangan jaringan menggunakan Cascading XGBoost Classifier")

st.sidebar.header("Input Network Traffic")

duration = st.sidebar.number_input("Duration", min_value=0, value=0)
protocol = st.sidebar.selectbox("Protocol Type", [0, 1, 2],
                                 format_func=lambda x: ['icmp','tcp','udp'][x])
src_bytes = st.sidebar.number_input("Source Bytes", min_value=0, value=181)
dst_bytes = st.sidebar.number_input("Destination Bytes", min_value=0, value=5450)
count = st.sidebar.number_input("Count", min_value=0, value=2)
srv_count = st.sidebar.number_input("Srv Count", min_value=0, value=2)
same_srv_rate = st.sidebar.slider("Same Srv Rate", 0.0, 1.0, 0.5)
diff_srv_rate = st.sidebar.slider("Diff Srv Rate", 0.0, 1.0, 0.5)
wrong_fragment = st.sidebar.number_input("Wrong Fragment", min_value=0, value=0)
serror_rate = st.sidebar.slider("Serror Rate", 0.0, 1.0, 0.0)

if st.sidebar.button("🔍 Deteksi Sekarang"):
    input_dict = {col: 0 for col in feature_names}
    input_dict['duration'] = duration
    input_dict['protocol_type'] = protocol
    input_dict['src_bytes'] = src_bytes
    input_dict['dst_bytes'] = dst_bytes
    input_dict['wrong_fragment'] = wrong_fragment
    input_dict['same_srv_rate'] = same_srv_rate
    input_dict['diff_srv_rate'] = diff_srv_rate
    input_dict['serror_rate'] = serror_rate
    input_dict['count'] = count
    input_dict['srv_count'] = srv_count

    input_df = pd.DataFrame([input_dict])

    # stage 1 — binary detection
    binary_pred = xgb_binary.predict(input_df)[0]
    binary_proba = xgb_binary.predict_proba(input_df)[0]

    st.subheader("Hasil Deteksi")
    
    st.write("Debug input:", input_dict['protocol_type'], 
         input_dict['count'], input_dict['src_bytes'])
    
    if binary_pred == 0:
        st.success(f"✅ Traffic NORMAL (confidence: {binary_proba[0]*100:.1f}%)")

        # confidence chart
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.barplot(x=['Normal', 'Anomali'], 
                   y=[binary_proba[0], binary_proba[1]],
                   palette=['green', 'red'], ax=ax)
        ax.set_ylabel('Probability')
        ax.set_title('Confidence Score')
        st.pyplot(fig)

    else:
        st.error(f"🚨 ANOMALI TERDETEKSI! (confidence: {binary_proba[1]*100:.1f}%)")

        # stage 2 — attack classification
        attack_pred = xgb_attack.predict(input_df)[0]
        attack_proba = xgb_attack.predict_proba(input_df)[0]
        attack_name = attack_label_names.get(attack_pred, "Unknown")

        st.warning(f"⚔️ Tipe Serangan: **{attack_name.upper()}**")

        # confidence chart attack
        fig, ax = plt.subplots(figsize=(10, 4))
        labels = [attack_label_names[i] for i in range(len(attack_proba))]
        sns.barplot(x=labels, y=attack_proba, palette='Reds_r', ax=ax)
        ax.set_ylabel('Probability')
        ax.set_title('Confidence per Tipe Serangan')
        plt.xticks(rotation=45)
        st.pyplot(fig)

else:
    st.info("👈 Isi parameter di sidebar kiri, lalu klik Deteksi Sekarang!")

    st.subheader("Tentang Project Ini")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Dataset", "KDD Cup 99")
    with col2:
        st.metric("Stage 1", "Binary XGBoost")
    with col3:
        st.metric("Stage 2", "Attack Classifier")
    with col4:
        st.metric("Accuracy", "99%+")

    st.subheader("Cara Kerja Cascading Classifier")
    st.markdown("""
    1. **Stage 1** — Model pertama deteksi apakah traffic Normal atau Anomali
    2. **Stage 2** — Kalau Anomali, model kedua identifikasi tipe serangannya
    """)