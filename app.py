# importing required dependencies
import streamlit as st
import numpy as np
import re
import joblib
import nltk

# downloading and importing stopwords
from nltk.corpus import stopwords

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# loading saved models
vectorizer = joblib.load("tfidf_vectorizer.joblib")
classifier = joblib.load("classifier.joblib")
regressor = joblib.load("regressor.joblib")

# function for cleaning text
def clean_text(text):
    if not isinstance(text, str) or text.strip() == "":
        text = "missing info"
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s+*/=<>-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    tokens = [word for word in text.split() if word not in stop_words]
    return " ".join(tokens)

# for extracting features
keywords = [
    "graph", "tree", "dp", "dynamic programming",
    "recursion", "greedy", "binary search",
    "sorting", "heap", "stack", "queue"
]

# number of keywords
def keyword_features(text):
    return sum(text.count(k) for k in keywords)

# strucutral features
def structural_features(text):
    return [
        len(text.split()),  # text length
        len(re.findall(r"[+\-*/=<>]", text)),  # math symbols
        text.count("if"),  # conditional hints
        text.count("for") + text.count("while"),  # loop hints
    ]


st.set_page_config(page_title="AutoJudge", layout="centered")
st.title("AutoJudge – Problem Difficulty Predictor")

st.write("Paste the problem details below:")

problem_desc = st.text_area("Problem Description")
input_desc = st.text_area("Input Description")
output_desc = st.text_area("Output Description")

if st.button("Predict Difficulty"):
    # cleans inputs
    problem_desc = clean_text(problem_desc)
    input_desc = clean_text(input_desc)
    output_desc = clean_text(output_desc)

    # combines text
    combined_text = " ".join([problem_desc, input_desc, output_desc])

    # tf-idf vector
    tfidf_vec = vectorizer.transform([combined_text]).toarray()

    # engineered features
    engineered_features = []
    engineered_features.append(keyword_features(combined_text))
    engineered_features.extend(structural_features(combined_text))

    engineered_features = np.array(engineered_features).reshape(1, -1)

    # combines tf-idf and engineered features
    final_vector = np.hstack([tfidf_vec, engineered_features])

    # predictions
    predicted_class = classifier.predict(final_vector)[0]
    predicted_score = regressor.predict(final_vector)[0]

    # display results
    st.success(f"Predicted Difficulty Class: **{predicted_class}**")
    st.info(f"Predicted Difficulty Score: **{round(predicted_score, 2)}**")