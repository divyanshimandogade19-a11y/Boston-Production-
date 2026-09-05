import joblib
import pandas as pd
import streamlit as st

# ---- Page setup ----
st.set_page_config(page_title="Boston House Price Predictor", page_icon="🏠")
st.title("🏠 Boston House Price Predictor")
st.write("Enter the property details below to predict the median house value (MEDV).")

# ---- Load model ----
@st.cache_resource
def load_model():
    return joblib.load("Boston_Project_Model.pkl")

model = load_model()

# ---- Input form ----
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        ID = st.number_input("ID", value=25, step=1)
        indus = st.number_input("INDUS (proportion of non-retail business acres)", value=7.07, format="%.3f")
        chas = st.selectbox("CHAS (bounds Charles River?)", options=[0, 1], index=0)
        nox = st.number_input("NOX (nitric oxide concentration)", value=0.469, format="%.3f")
        rm = st.number_input("RM (avg number of rooms)", value=6.421, format="%.3f")
        age = st.number_input("AGE (% built before 1940)", value=78.9, format="%.1f")

    with col2:
        dis = st.number_input("DIS (distance to employment centres)", value=4.967, format="%.3f")
        rad = st.number_input("RAD (accessibility to highways index)", value=2, step=1)
        tax = st.number_input("TAX (property tax rate)", value=242, step=1)
        ptratio = st.number_input("PTRATIO (pupil-teacher ratio)", value=17.8, format="%.1f")
        lstat = st.number_input("LSTAT (% lower status population)", value=9.14, format="%.2f")

    submitted = st.form_submit_button("Predict")

# ---- Predict ----
if submitted:
    new_data = pd.DataFrame({
        "ID": [ID],
        "indus": [indus],
        "chas": [chas],
        "nox": [nox],
        "rm": [rm],
        "age": [age],
        "dis": [dis],
        "rad": [rad],
        "tax": [tax],
        "ptratio": [ptratio],
        "lstat": [lstat],
    })

    prediction = model.predict(new_data)[0]
    st.success(f"### Predicted MEDV: **{prediction:.2f}**  (in $1000s)")