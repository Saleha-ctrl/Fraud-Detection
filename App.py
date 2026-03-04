import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===============================
# ⚙️ PAGE CONFIGURATION
# ===============================
st.set_page_config(
    page_title="AI Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

# ===============================
# 🎨 CUSTOM STYLING
# ===============================
st.markdown("""
<style>
/* Background Gradient */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 80%);
    color: #f8fafc;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e3a8a 0%, #0f172a 100%);
    color: white;
}
[data-testid="stSidebar"] h2 {
    color: #a5f3fc !important;
}
[data-testid="stSidebar"] .stRadio label {
    color: #f8fafc !important;
    font-weight: 500;
}

/* Header */
.header {
    text-align: center;
    background: linear-gradient(90deg, #1e3a8a 0%, #0f172a 100%);
    border-radius: 20px;
    padding: 25px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    margin-bottom: 25px;
}
.header h1 {
    color: #ffffff;
    font-weight: 800;
    font-size: 38px;
    margin-bottom: 5px;
}
.header p {
    color: #c7d2fe;
    font-size: 16px;
}

/* Section Titles */
.section-title {
    font-size: 22px;
    font-weight: 700;
    color: #bfdbfe;
    margin-top: 30px;
    margin-bottom: 10px;
}

/* Stat Cards */
.stat-card {
    background-color: rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    text-align: center;
    padding: 18px;
    color: #ffffff;
    box-shadow: 0 3px 8px rgba(0,0,0,0.25);
}
.stat-card h2 {
    font-size: 28px;
    margin: 0;
    color: #ffffff;
}
.stat-card p {
    color: #a5f3fc;
    margin-top: 4px;
}

/* File uploader */
[data-testid="stFileUploader"] > label div {
    color: #ffffff !important;
    font-weight: 600 !important;
}
[data-testid="stFileUploader"] section {
    background-color: rgba(255,255,255,0.1) !important;
    border: 2px dashed #60a5fa !important;
    border-radius: 12px !important;
}

/* Buttons */
.stButton>button {
    background-color: #3b82f6;
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 10px 20px;
    transition: all 0.3s ease;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
.stButton>button:hover {
    background-color: #1d4ed8;
    transform: scale(1.05);
}

/* Result Boxes */
.result-box {
    text-align: center;
    font-size: 22px;
    font-weight: 700;
    padding: 20px;
    border-radius: 12px;
    margin-top: 20px;
}
.fraud {
    background-color: #dc2626;
    color: white;
}
.legit {
    background-color: #16a34a;
    color: white;
}

/* Footer */
.footer {
    text-align: center;
    color: #cbd5e1;
    margin-top: 40px;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# 📦 LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    return joblib.load("fraud_model.pkl")

model = load_model()

# ===============================
# 🧭 SIDEBAR NAVIGATION
# ===============================
st.sidebar.title("💳 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "🔍 Single Prediction", "📁 Batch Analysis", "ℹ️ About Project"]
)

# ===============================
# 🏠 HOME PAGE
# ===============================
if page == "🏠 Home":
    st.markdown("""
    <div class='header'>
        <h1>💳 AI Fraud Detection Dashboard</h1>
        <p>Detect suspicious credit card transactions using Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 🌟 Overview
    This project uses **Logistic Regression** and **Decision Tree Classifiers** to detect fraudulent credit card transactions.  
    The dataset is highly imbalanced, so **SMOTE (Synthetic Minority Oversampling Technique)** was used to balance it.

    #### 🔍 Features:
    - Real-time prediction for single transactions  
    - Batch CSV file analysis  
    - Visual performance metrics  
    - Elegant UI powered by Streamlit  

    #### 💡 Technologies Used:
    - Python 🐍  
    - Scikit-learn  
    - Pandas, NumPy  
    - Streamlit (Frontend UI)  
    - Matplotlib (Visualization)
    """)

# ===============================
# 🔍 SINGLE TRANSACTION PAGE
# ===============================
elif page == "🔍 Single Prediction":
    st.markdown("<h3 class='section-title'>🧾 Single Transaction Prediction</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        v1 = st.number_input("Feature V1", value=0.0, step=0.1)
        v2 = st.number_input("Feature V2", value=0.0, step=0.1)
    with col2:
        v3 = st.number_input("Feature V3", value=0.0, step=0.1)
        amount = st.number_input("Transaction Amount", value=0.0, step=1.0)

    if st.button("🔍 Predict Transaction"):
        features = np.zeros((1, 30))
        features[0, 1] = v1
        features[0, 2] = v2
        features[0, 3] = v3
        features[0, -1] = amount

        pred = model.predict(features)[0]

        if pred == 1:
            st.markdown("<div class='result-box fraud'>🚨 Fraudulent Transaction Detected!</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='result-box legit'>✅ Legitimate Transaction</div>", unsafe_allow_html=True)

# ===============================
# 📁 BATCH ANALYSIS PAGE
# ===============================
elif page == "📁 Batch Analysis":
    st.markdown("<h3 class='section-title'>📁 Batch Transaction Analysis</h3>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload your transaction CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("📊 **Preview of Uploaded Data:**")
        st.dataframe(df.head())

        preds = model.predict(df)
        df["Prediction"] = ["🚨 Fraud" if p == 1 else "✅ Legit" for p in preds]

        fraud_count = np.sum(preds)
        legit_count = len(preds) - fraud_count

        # Dashboard Stats
        st.markdown("<h3 class='section-title'>📈 Summary</h3>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"<div class='stat-card'><h2>{len(preds)}</h2><p>Total</p></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='stat-card'><h2 style='color:#fca5a5'>{fraud_count}</h2><p>Fraud</p></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='stat-card'><h2 style='color:#86efac'>{legit_count}</h2><p>Legit</p></div>", unsafe_allow_html=True)

        def highlight_fraud(val):
            color = '#dc2626' if val == '🚨 Fraud' else '#16a34a'
            return f'color: white; background-color: {color}; font-weight: bold;'

        st.write("🔎 **Predictions (Top 20 Rows):**")
        st.dataframe(df.head(20).style.applymap(highlight_fraud, subset=["Prediction"]))

        csv_download = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Predictions",
            data=csv_download,
            file_name="fraud_predictions.csv",
            mime="text/csv",
        )

# ===============================
# ℹ️ ABOUT PAGE
# ===============================
elif page == "ℹ️ About Project":
    st.markdown("<h3 class='section-title'>ℹ️ About This Project</h3>", unsafe_allow_html=True)
    st.write("""
    This **AI Fraud Detection System** is developed by **Saleha**  
    as part of a Machine Learning course project.  

    #### 🚀 Purpose:
    To detect fraudulent transactions using Machine Learning algorithms,  
    providing businesses with a real-time fraud analysis tool.

    #### 🧠 Model Info:
    - Logistic Regression (balanced using SMOTE)
    - Decision Tree Classifier for comparison  
    - Evaluated using Accuracy, Precision, Recall, F1, AUC  

    #### 📅 Year:
    2025

    #### 🧑‍💻 Developer:
    **Saleha Asif**  
    Department of Software Engineering  
    """)

st.markdown("<div class='footer'>💠 Developed by <b>Saleha</b> — AI & ML Project (2025)</div>", unsafe_allow_html=True)
