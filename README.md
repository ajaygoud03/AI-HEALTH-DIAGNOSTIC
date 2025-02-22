# AI-HEALTH-DIAGNOSTIC
AI-Based Health Diagnostic Assistant 🔹 Idea: A chatbot that predicts diseases based on symptoms using ML. 🔹 Tech Stack: Python, TensorFlow, NLP, Flask. 🔹 Steps:  Train a model on disease-symptom datasets. Create an AI-powered chatbot for user queries. Provide probability-based disease predictions.
1. Python Backend Review
✅ Strengths:
Proper Use of Pandas & NumPy: Efficient data handling and preprocessing.
ML Model with TensorFlow/Keras: Uses a multi-layer perceptron (MLP) model to classify diseases based on symptoms.
Data Splitting: Implements train_test_split correctly to prevent data leakage.
Flask API Implementation: Well-structured API routes (/ and /predict).
Encoders Used: LabelEncoder and MultiLabelBinarizer for categorical encoding
2. HTML Frontend Review
✅ Strengths:
Clean and user-friendly UI.
Minimal CSS for styling, making it lightweight.
JavaScript Fetch API correctly sends symptom data to the Flask API.
Uses event listeners effectively to capture user input
3. Streamlit UI Review
✅ Strengths:
Simple & Interactive UI
Uses requests.post() to fetch API results.
Error handling is implemented (st.warning).
4. Dataset Review
✅ Strengths:
CSV file contains Disease and Symptom columns.
Some preprocessing is done before model training.