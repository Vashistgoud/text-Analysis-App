import streamlit as st
import numpy as np
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Define stopwords
sw = nltk.corpus.stopwords.words("english")

# Streamlit Sidebar
rad = st.sidebar.radio("Navigation", ["Home", "Spam or Ham Detection", "Sentiment Analysis", "Stress Detection", "Hate and Offensive Content Detection", "Sarcasm Detection"])

# Function to clean and transform the user input
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [i for i in text if i.isalnum()]
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]
    ps = PorterStemmer()
    text = [ps.stem(i) for i in text]
    return " ".join(text)

# Load and preprocess data
def load_and_preprocess_data(filename, encoding='ISO-8859-1'):
    try:
        df = pd.read_csv(filename, encoding=encoding)
        return df
    except Exception as e:
        st.error(f"An error occurred while reading {filename}: {e}")
        return pd.DataFrame()

# Spam Detection
df1 = load_and_preprocess_data("Spam Detection.csv")
df1.columns = ["Label", "Text"]
tfidf1 = TfidfVectorizer(stop_words=sw, max_features=20)
x1 = tfidf1.fit_transform(df1["Text"]).toarray()
y1 = df1["Label"]
x_train1, x_test1, y_train1, y_test1 = train_test_split(x1, y1, test_size=0.1, random_state=0)
model1 = LogisticRegression()
model1.fit(x_train1, y_train1)

# Sentiment Analysis
df2 = load_and_preprocess_data("Sentiment Analysis.csv")
df2.columns = ["Text", "Label"]
tfidf2 = TfidfVectorizer(stop_words=sw, max_features=20)
x2 = tfidf2.fit_transform(df2["Text"]).toarray()
y2 = df2["Label"]
x_train2, x_test2, y_train2, y_test2 = train_test_split(x2, y2, test_size=0.1, random_state=0)
model2 = LogisticRegression()
model2.fit(x_train2, y_train2)

# Stress Detection
df3 = load_and_preprocess_data("Stress Detection.csv")
df3 = df3.drop(["subreddit", "post_id", "sentence_range", "syntax_fk_grade"], axis=1)
df3.columns = ["Text", "Sentiment", "Stress Level"]
tfidf3 = TfidfVectorizer(stop_words=sw, max_features=20)
x3 = tfidf3.fit_transform(df3["Text"]).toarray()
y3 = df3["Stress Level"].to_numpy()
x_train3, x_test3, y_train3, y_test3 = train_test_split(x3, y3, test_size=0.1, random_state=0)
model3 = DecisionTreeRegressor(max_leaf_nodes=2000)
model3.fit(x_train3, y_train3)

# Hate and Offensive Content Detection
df4 = load_and_preprocess_data("Hate Content Detection.csv")
df4 = df4.drop(["Unnamed: 0", "count", "neither"], axis=1)
df4.columns = ["Hate Level", "Offensive Level", "Class Level", "Text"]
tfidf4 = TfidfVectorizer(stop_words=sw, max_features=20)
x4 = tfidf4.fit_transform(df4["Text"]).toarray()
y4 = df4["Class Level"]
x_train4, x_test4, y_train4, y_test4 = train_test_split(x4, y4, test_size=0.1, random_state=0)
model4 = RandomForestClassifier()
model4.fit(x_train4, y_train4)

# Sarcasm Detection
df5 = load_and_preprocess_data("Sarcasm Detection.csv")
df5.columns = ["Text", "Label"]
tfidf5 = TfidfVectorizer(stop_words=sw, max_features=20)
x5 = tfidf5.fit_transform(df5["Text"]).toarray()
y5 = df5["Label"]
x_train5, x_test5, y_train5, y_test5 = train_test_split(x5, y5, test_size=0.1, random_state=0)
model5 = LogisticRegression()
model5.fit(x_train5, y_train5)

# Home Page
if rad == "Home":
    st.title("Complete Text Analysis App")
    st.image("Complete Text Analysis Home Page.jpg")
    st.text(" ")
    st.text("The Following Text Analysis Options Are Available->")
    st.text(" ")
    st.text("1. Spam or Ham Detection")
    st.text("2. Sentiment Analysis")
    st.text("3. Stress Detection")
    st.text("4. Hate and Offensive Content Detection")
    st.text("5. Sarcasm Detection")

# Spam or Ham Detection Page
if rad == "Spam or Ham Detection":
    st.header("Detect Whether A Text Is Spam Or Ham")
    sent1 = st.text_area("Enter The Text")
    if sent1:
        transformed_sent1 = transform_text(sent1)
        vector_sent1 = tfidf1.transform([transformed_sent1])
        if st.button("Predict"):
            prediction1 = model1.predict(vector_sent1)[0]
            if prediction1 == "spam":
                st.warning("Spam Text!!")
            elif prediction1 == "ham":
                st.success("Ham Text!!")

# Sentiment Analysis Page
if rad == "Sentiment Analysis":
    st.header("Detect The Sentiment Of The Text")
    sent2 = st.text_area("Enter The Text")
    if sent2:
        transformed_sent2 = transform_text(sent2)
        vector_sent2 = tfidf2.transform([transformed_sent2])
        if st.button("Predict"):
            prediction2 = model2.predict(vector_sent2)[0]
            if prediction2 == 0:
                st.warning("Negative Text!!")
            elif prediction2 == 1:
                st.success("Positive Text!!")

# Stress Detection Page
if rad == "Stress Detection":
    st.header("Detect The Amount Of Stress In The Text")
    sent3 = st.text_area("Enter The Text")
    if sent3:
        transformed_sent3 = transform_text(sent3)
        vector_sent3 = tfidf3.transform([transformed_sent3])
        if st.button("Predict"):
            prediction3 = model3.predict(vector_sent3)[0]
            if prediction3 >= 0:
                st.warning("Stressful Text!!")
            else:
                st.success("Not A Stressful Text!!")

# Hate and Offensive Content Detection Page
if rad == "Hate and Offensive Content Detection":
    st.header("Detect The Level Of Hate & Offensive Content In The Text")
    sent4 = st.text_area("Enter The Text")
    if sent4:
        transformed_sent4 = transform_text(sent4)
        vector_sent4 = tfidf4.transform([transformed_sent4])
        if st.button("Predict"):
            prediction4 = model4.predict(vector_sent4)[0]
            if prediction4 == 0:
                st.exception("Highly Offensive Text!!")
            elif prediction4 == 1:
                st.warning("Offensive Text!!")
            elif prediction4 == 2:
                st.success("Non-Offensive Text!!")

# Sarcasm Detection Page
if rad == "Sarcasm Detection":
    st.header("Detect Whether The Text Is Sarcastic Or Not")
    sent5 = st.text_area("Enter The Text")
    if sent5:
        transformed_sent5 = transform_text(sent5)
        vector_sent5 = tfidf5.transform([transformed_sent5])
        if st.button("Predict"):
            prediction5 = model5.predict(vector_sent5)[0]
            if prediction5 == 1:
                st.exception("Sarcastic Text!!")
            elif prediction5 == 0:
                st.success("Non-Sarcastic Text!!")
