import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Page Title
st.set_page_config(page_title="Credit Scoring Model", layout="centered")
st.title("💳 Credit Scoring Model")
st.header("Machine Learning Project using Logistic Regression")


#load dataset
st.sidebar.title("⚙️ Controls")
st.sidebar.write("Upload a CSV file to train the model.")
# Upload CSV
uploaded_file = st.sidebar.file_uploader(
    "Upload csv file",
    type=["csv"]
)

if uploaded_file is not None:
    
    df = pd.read_csv(uploaded_file)

    st.success("Dataset uploaded successfully!")
    st.divider()
    
    with st.expander("📂 View Dataset Preview"):
        st.dataframe(df.head()) 
    st.divider()
  
    
    st.subheader("📈 Statistical Summary")
    st.dataframe(df.describe())
    st.divider()
  
    
    st.subheader("📋 Dataset Information")
  
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", df.isnull().sum().sum())
    st.divider()
  
    
    st.subheader("📊 Age Distribution")

    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(df["Age"], bins=15, edgecolor="black")
    ax.set_title("Customer Age Distribution")
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    st.divider()
    
    
    st.subheader("💰 Credit Amount Distribution")

    fig, ax = plt.subplots(figsize=(6,4))

    ax.hist(df["Credit amount"], bins=20, edgecolor="black")

    ax.set_xlabel("Credit Amount")
    ax.set_ylabel("Customers")

    st.pyplot(fig)
    st.divider()
  
    
    st.subheader("🥧 Credit Risk Distribution")

    risk = df["Credit Risk"].value_counts()

    fig, ax = plt.subplots(figsize=(5,5))

    ax.pie(
        risk,
        labels=["Good","Bad"],
        autopct="%1.1f%%",
        startangle=90
    )

    st.pyplot(fig)
    
    #features(input)
    X = df.drop(["Unnamed: 0","Credit Risk"], axis=1)

    #output
    Y=df["Credit Risk"]


    # Convert all text columns to numbers
    le = LabelEncoder()

    for column in X.select_dtypes(include="object").columns:
        X[column] = le.fit_transform(X[column].astype(str))

        # Convert target column if it contains text
    Y = LabelEncoder().fit_transform(Y)
  

    #split dataset into training and testing
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42) 


    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    #create the model
    model=LogisticRegression(max_iter=1000)

    #train the model
    model.fit(X_train,Y_train)

    #predict on testd data
    pred=model.predict(X_test)

    #Results
    # ----------------------------
    if st.sidebar.button("Train Model & Show Results"):
        accuracy = accuracy_score(Y_test, pred)
        precision = precision_score(Y_test, pred)
        recall = recall_score(Y_test, pred)
        f1 = f1_score(Y_test, pred)
    

        st.success("Model Trained Successfully!")
        st.balloons()
        
        st.subheader("📈 Model Performance")

        scores = [accuracy, precision, recall, f1]
        labels = ["Accuracy", "Precision", "Recall", "F1 Score"]

        fig, ax = plt.subplots(figsize=(6,4))

        ax.bar(labels, scores)

        ax.set_ylim(0,1)

        ax.set_ylabel("Score")

        st.pyplot(fig)
   
        with st.container():
            col1, col2 = st.columns(2)

            col1.metric("Accuracy", f"{accuracy*100:.2f}%")
            col1.metric("Precision", f"{precision*100:.2f}%")

            col2.metric("Recall", f"{recall*100:.2f}%")
            col2.metric("F1 Score", f"{f1*100:.2f}%")
        st.divider()

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(Y_test, pred)

        fig, ax = plt.subplots(figsize=(5,4))

        im = ax.imshow(cm, cmap="Blues")
        plt.colorbar(im)

        ax.set_xticks(np.arange(2))
        ax.set_yticks(np.arange(2))

        ax.set_xticklabels(["Good", "Bad"])
        ax.set_yticklabels(["Good", "Bad"])

        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j],
                    ha="center",
                    va="center",fontsize=14,color="white" if cm[i,j] > cm.max()/2 else "black")

        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")

        st.pyplot(fig)
        st.divider()

        st.subheader("Classification Report")
        st.text(classification_report(Y_test, pred))
        
        
        
        st.divider()
        st.subheader("📌 Feature Importance")

        importance = pd.DataFrame({
            "Feature": X.columns,
            "Coefficient": model.coef_[0]
        })

        importance = importance.sort_values("Coefficient")

        fig, ax = plt.subplots(figsize=(8,5))

        ax.barh(
            importance["Feature"],
            importance["Coefficient"]
        )

        ax.set_xlabel("Coefficient")

        st.pyplot(fig)
        st.markdown("---")
        st.caption("Developed using Streamlit • Scikit-learn • Matplotlib")



