import streamlit as st
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Title
st.title("Sonar Rock vs Mine Predictor")

# Load dataset
df = pd.read_csv("sonar.csv", header=None)
df[60] = df[60].map({'R': 0, 'M': 1})

# Split
X = df.drop(60, axis=1)
y = df[60]

# Scale
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train model
model = SVC()
model.fit(X, y)

st.success("Model trained successfully")

# Input section
st.subheader("Enter Sonar Values")

input_data = []
for i in range(60):
    value = st.slider(f"Feature {i+1}", 0.0, 1.0, 0.5)
    input_data.append(value)

# Prediction
if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)

    if prediction[0] == 0:
        st.success("Result: Rock")
    else:
        st.error("Result: Mine")
        