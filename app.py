import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Title of the App
st.title("Machine Learning Model GUI")

# Sidebar - Upload File
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Dataset")
    st.write(df.head())

    # Select target column
    target_column = st.sidebar.selectbox("Select Target Column", df.columns)

    if target_column:
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Convert categorical data
        X = pd.get_dummies(X, drop_first=True)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardization
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        st.sidebar.write("✅ Data Preprocessed Successfully!")

        # Select algorithm
        algorithm = st.sidebar.selectbox("Select Algorithm", ["Decision Tree", "Naive Bayes", "KNN", "SVM", "Logistic Regression", "Linear Regression"])

        # Apply selected algorithm
        if algorithm == "Decision Tree":
            model = DecisionTreeClassifier()
        elif algorithm == "Naive Bayes":
            model = GaussianNB()
        elif algorithm == "KNN":
            model = KNeighborsClassifier(n_neighbors=5)
        elif algorithm == "SVM":
            model = SVC(kernel='linear')
        elif algorithm == "Logistic Regression":
            model = LogisticRegression()
        elif algorithm == "Linear Regression":
            model = LinearRegression()

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        # Show results
        st.write(f"### {algorithm} Results")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

        # Confusion Matrix for classifiers
        if algorithm != "Linear Regression":
            cm = confusion_matrix(y_test, y_pred)
            st.write("Confusion Matrix:")
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
            st.pyplot(plt)

        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))
