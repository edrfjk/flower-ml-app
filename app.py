import streamlit as st
import joblib
from sklearn.datasets import load_iris
import numpy as np

# Load model
model = joblib.load('flower_model.joblib')
iris = load_iris()

# Page config
st.set_page_config(page_title="Flower ML App", page_icon="🌸", layout="wide")


st.markdown("""
<style>
.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #4CAF50;
}
.subtitle {
    text-align: center;
    color: #aaa;
    margin-bottom: 30px;
}
.box {
    padding: 20px;
    border-radius: 15px;
    background: #111;
    border: 1px solid #2a2a2a;
}
.result-box {
    padding: 30px;
    border-radius: 15px;
    background: linear-gradient(135deg, #1b5e20, #2e7d32);
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# -------- HEADER --------
st.markdown('<div class="title">🌸 Iris Flower Classification System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict flower species using ML</div>', unsafe_allow_html=True)


col1, col2 = st.columns([1, 1])


with col1:
    st.markdown("### 🧾 Input Features")

    sepal_length = st.number_input("Sepal Length", 0.0, 10.0, 5.1)
    sepal_width = st.number_input("Sepal Width", 0.0, 10.0, 3.5)
    petal_length = st.number_input("Petal Length", 0.0, 10.0, 1.4)
    petal_width = st.number_input("Petal Width", 0.0, 10.0, 0.2)

    predict_btn = st.button("🔍 Predict")


with col2:
    st.markdown("### 🌼 Prediction Result")

    if predict_btn:
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(input_data)
        result = iris.target_names[prediction][0]

        st.markdown(f"""
        <div class="result-box">
            <h2 style="color:white;">{result.upper()}</h2>
            <p style="color:#c8e6c9;">Predicted Flower</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.info("Enter values and click Predict")