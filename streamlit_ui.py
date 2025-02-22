import streamlit as st
import requests

# Streamlit UI
st.title("🩺 AI-Based Health Diagnostic Assistant")
st.write("Enter your symptoms below and get a probable diagnosis.")

# User input
symptoms_input = st.text_input("Enter symptoms (comma-separated):")

def get_prediction(symptoms):
    api_url = "http://127.0.0.1:5000/predict"  # Update with deployed API URL if needed
    response = requests.post(api_url, json={"symptoms": symptoms})
    if response.status_code == 200:
        return response.json().get("predicted_disease", "No diagnosis available.")
    return "Error: Could not fetch diagnosis."

if st.button("Diagnose"):
    if symptoms_input:
        symptoms_list = [s.strip() for s in symptoms_input.split(",")]
        diagnosis = get_prediction(symptoms_list)
        st.success(f"🩺 Predicted Disease: {diagnosis}")
    else:
        st.warning("Please enter at least one symptom.")
