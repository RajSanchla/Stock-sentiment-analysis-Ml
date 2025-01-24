import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Title and Description
st.title("Stock Sentiment Analysis")
st.write("This application performs sentiment analysis on stock market-related data.")

# Upload Data
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.write(data.head())

    # Display dataset statistics
    st.write("### Dataset Information")
    st.write(data.describe())

    # Text Preprocessing
    st.write("### Text Preprocessing")
    text_column = st.text_input("Enter the column name for text data:", "text")
    if text_column in data.columns:
        st.write(f"Selected Text Column: {text_column}")

        # Sentiment Label Column
        sentiment_column = st.text_input("Enter the column name for sentiment labels:", "sentiment")
        if sentiment_column in data.columns:
            st.write(f"Selected Sentiment Column: {sentiment_column}")

            # Text Vectorization
            st.write("### Vectorizing Text Data")
            vectorizer = CountVectorizer()
            X = vectorizer.fit_transform(data[text_column].fillna(""))
            y = data[sentiment_column]

            # Train-Test Split
            st.write("### Splitting Data into Train and Test Sets")
            test_size = st.slider("Select Test Size Ratio:", 0.1, 0.5, 0.2)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            # Model Training
            st.write("### Training the Model")
            model = MultinomialNB()
            model.fit(X_train, y_train)
            
            # Model Evaluation
            st.write("### Model Evaluation")
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Accuracy: {accuracy:.2f}")
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

            # Prediction on New Input
            st.write("### Make Predictions")
            user_input = st.text_area("Enter a sentence to analyze sentiment:")
            if user_input:
                input_vectorized = vectorizer.transform([user_input])
                prediction = model.predict(input_vectorized)
                st.write(f"Predicted Sentiment: {prediction[0]}")

            # Visualization
            st.write("### Data Visualization")
            sentiment_counts = data[sentiment_column].value_counts()
            fig, ax = plt.subplots()
            sentiment_counts.plot(kind='bar', ax=ax, color=['blue', 'orange'])
            plt.title("Sentiment Distribution")
            plt.xlabel("Sentiment")
            plt.ylabel("Count")
            st.pyplot(fig)

        else:
            st.error(f"Column '{sentiment_column}' not found in the dataset.")
    else:
        st.error(f"Column '{text_column}' not found in the dataset.")
else:
    st.info("Please upload a CSV file to get started.")
