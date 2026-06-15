import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# load model dan mapping
xgb_model = joblib.load('model/xgb_model.pkl')
label_to_int = joblib.load('model/label_to_int.pkl')
int_to_label = joblib.load('model/int_to_label.pkl')
feature_names = joblib.load('model/feature_names.pkl')

attack_names = {
    0: 'back', 1: 'ftp_write', 2: 'ipsweep',
    3: 'neptune', 4: 'normal', 5: 'nmap',
    6: 'portsweep', 7: 'satan', 8: 'smurf',
    9: 'warezclient'
}

st.set_page_config(
    page_title="Network Anomaly Detection",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Network Anomaly Detection")
st.markdown("Deteksi serangan jaringan menggunakan Machine Learning (XGBoost)")

st.sidebar.header("Input Network Traffic")

duration = st.sidebar.number_input("Duration", min_value=0, value=0)
protocol = st.sidebar.selectbox("Protocol Type", [0, 1, 2],
                                 format_func=lambda x: ['tcp','udp','icmp'][x])
src_bytes = st.sidebar.number_input("Source Bytes", min_value=0, value=181)
dst_bytes = st.sidebar.number_input("Destination Bytes", min_value=0, value=5450)
count = st.sidebar.number_input("Count", min_value=0, value=9)
srv_count = st.sidebar.number_input("Srv Count", min_value=0, value=9)
same_srv_rate = st.sidebar.slider("Same Srv Rate", 0.0, 1.0, 1.0)
diff_srv_rate = st.sidebar.slider("Diff Srv Rate", 0.0, 1.0, 0.0)
wrong_fragment = st.sidebar.number_input("Wrong Fragment", min_value=0, value=0)
serror_rate = st.sidebar.slider("Serror Rate", 0.0, 1.0, 0.0)

if st.sidebar.button("🔍 Deteksi Sekarang"):
    # buat input dengan nama kolom yang bener
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
    prediction = xgb_model.predict(input_df)[0]
    proba = xgb_model.predict_proba(input_df)[0]

    attack_label = attack_names.get(prediction, f"Unknown ({prediction})")

    st.subheader("Hasil Deteksi")

    if attack_label == 'normal':
        st.success(f"✅ Traffic NORMAL — Tidak terdeteksi serangan")
    else:
        st.error(f"🚨 SERANGAN TERDETEKSI: **{attack_label.upper()}**")

    st.subheader("Confidence Score")
    proba_df = pd.DataFrame({
        'Tipe': [attack_names.get(i, str(i)) for i in range(len(proba))],
        'Probability': proba
    }).sort_values('Probability', ascending=False).head(5)

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=proba_df, x='Tipe', y='Probability', ax=ax)
    ax.set_title('Top 5 Prediksi')
    st.pyplot(fig)

else:
    st.info("👈 Isi parameter di sidebar kiri, lalu klik Deteksi Sekarang!")

    st.subheader("Tentang Project Ini")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Dataset", "KDD Cup 99")
    with col2:
        st.metric("Model", "XGBoost")
    with col3:
        st.metric("Accuracy", "99%+")