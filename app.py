import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ===== Load Data =====
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 0
true["label"] = 1

df = pd.concat([fake, true])
df = df[["text", "label"]].dropna()

# ===== Train Model =====
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X = tfidf.fit_transform(df["text"])
y = df["label"]

model = LogisticRegression()
model.fit(X, y)

# ===== Streamlit UI =====
st.title("📰 Fake News Detector")

text = st.text_area("Enter news text here:")

if st.button("Predict"):
    if text.strip() != "":
        vector = tfidf.transform([text])
        pred = model.predict(vector)[0]

        if pred == 0:
            st.error("❌ Fake News")
        else:
            st.success("✅ Real News")
    else:
        st.warning("Please enter text") 