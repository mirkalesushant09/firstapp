import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Insurance Prediction",
    page_icon="🏥",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #667eea, #764ba2);
}

.main-title {
    text-align:center;
    color:white;
    font-size:45px;
    font-weight:bold;
    margin-bottom:20px;
}

.card {
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 5px 20px rgba(0,0,0,0.2);
    transition:0.3s;
}

.card:hover {
    transform:translateY(-5px);
}

.metric-card {
    background: linear-gradient(135deg,#11998e,#38ef7d);
    padding:20px;
    border-radius:15px;
    color:white;
    text-align:center;
    font-size:22px;
    font-weight:bold;
}

.stButton>button {
    width:100%;
    height:50px;
    border-radius:10px;
    border:none;
    background:#4CAF50;
    color:white;
    font-size:18px;
    font-weight:bold;
}

.stButton>button:hover {
    background:#45a049;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown(
    "<div class='main-title'>🏥 Insurance Purchase Prediction</div>",
    unsafe_allow_html=True
)

# ---------------- DATASET ----------------
df = pd.DataFrame({
    'age': [21, 48, 32, 41, 20, 35, 20, 23],
    'bought_insurance': [0, 1, 1, 1, 0, 1, 0, 0]
})

# ---------------- TRAIN MODEL ----------------
X = df[['age']]
y = df['bought_insurance']

model = LogisticRegression()
model.fit(X, y)

predictions = model.predict(X)
accuracy = accuracy_score(y, predictions)

# ---------------- LAYOUT ----------------
col1, col2 = st.columns([1, 1])

# ================= LEFT COLUMN =================
with col1:

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("📋 Enter Details")

    age = st.number_input(
        "Enter Age",
        min_value=1,
        max_value=100,
        value=25
    )

    if st.button("🔍 Check Insurance Eligibility"):

        prediction = model.predict([[age]])[0]
        probability = model.predict_proba([[age]])[0][1]

        st.write(f"### Age Entered: {age}")

        if prediction == 1:
            st.success("✅ Likely to BUY Insurance")
        else:
            st.error("❌ Likely NOT to Buy Insurance")

        st.progress(float(probability))

        st.info(
            f"Probability of Buying Insurance: {probability*100:.2f}%"
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ================= RIGHT COLUMN =================
with col2:

    st.markdown(
        f"""
        <div class='metric-card'>
            Model Accuracy<br>
            {accuracy*100:.2f}%
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    st.subheader("📊 Dataset")

    display_df = df.copy()
    display_df['bought_insurance'] = display_df[
        'bought_insurance'
    ].replace({0: "No", 1: "Yes"})

    st.dataframe(display_df, use_container_width=True)

# ---------------- SCATTER PLOT ----------------
st.write("")
st.subheader("📈 Age vs Insurance Purchase")

fig, ax = plt.subplots(figsize=(8, 4))

ax.scatter(
    df['age'],
    df['bought_insurance'],
    s=150
)

ax.set_xlabel("Age")
ax.set_ylabel("Bought Insurance")
ax.set_title("Insurance Purchase Distribution")

st.pyplot(fig)

# ---------------- MODEL INFO ----------------
with st.expander("ℹ️ About Model"):

    st.write("""
    This application uses Logistic Regression
    to predict whether a person is likely
    to purchase insurance based on age.

    Predictions:
    - 0 = Not Buy Insurance
    - 1 = Buy Insurance
    """)

# ---------------- FOOTER ----------------
st.markdown("""
<hr>
<center>
Made with ❤️ using Sushant,Streamlit & Scikit-Learn
</center>
""", unsafe_allow_html=True)