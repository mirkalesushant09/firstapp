import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Insurance Prediction",
    page_icon="🏥",
    layout="wide"
)

# ---------------- DATASET ----------------
df = pd.DataFrame({
    'age': [21, 48, 32, 41, 20, 35, 20, 23],
    'bought_insurance': [0, 1, 1, 1, 0, 1, 0, 0]
})

X = df[['age']]
y = df['bought_insurance']

# ---------------- PICKLE FILE ----------------
MODEL_FILE = "insurance_model.pkl"

# Create pickle file if it doesn't exist
if not os.path.exists(MODEL_FILE):

    model = LogisticRegression()
    model.fit(X, y)

    with open(MODEL_FILE, "wb") as file:
        pickle.dump(model, file)

# Load model from pickle file
with open(MODEL_FILE, "rb") as file:
    model = pickle.load(file)

# ---------------- MODEL ACCURACY ----------------
predictions = model.predict(X)
accuracy = accuracy_score(y, predictions)

# ---------------- TITLE ----------------
st.title("🏥 Insurance Purchase Prediction")

# ---------------- INPUT ----------------
age = st.number_input(
    "Enter Age",
    min_value=1,
    max_value=100,
    value=25
)

# ---------------- PREDICTION ----------------
if st.button("🔍 Check Insurance Eligibility"):

    user_data = [[age]]

    prediction = model.predict(user_data)[0]
    probability = model.predict_proba(user_data)[0][1]

    st.write(f"### Age Entered: {age}")

    if prediction == 1:
        st.success("✅ Likely to BUY Insurance")
    else:
        st.error("❌ Likely NOT to Buy Insurance")

    st.progress(float(probability))
    st.info(
        f"Probability of Buying Insurance: {probability * 100:.2f}%"
    )

# ---------------- ACCURACY ----------------
st.metric(
    "Model Accuracy",
    f"{accuracy * 100:.2f}%"
)

# ---------------- DATASET ----------------
st.subheader("📊 Dataset")

display_df = df.copy()
display_df["bought_insurance"] = display_df[
    "bought_insurance"
].replace({
    0: "No",
    1: "Yes"
})

st.dataframe(display_df, use_container_width=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.write("Made with ❤️ using Streamlit, Pickle and Scikit-Learn")