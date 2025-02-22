import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from flask import Flask, request, jsonify
import joblib

# Load dataset
df = pd.read_csv("disease_symptom_dataset.csv")

# Preprocessing
symptoms = df.drop(columns=['disease'])
diseases = df['disease']
le = LabelEncoder()
diseases_encoded = le.fit_transform(diseases)
mlb = MultiLabelBinarizer()
symptoms_encoded = mlb.fit_transform(symptoms.values)

# Split data
X_train, X_test, y_train, y_test = train_test_split(symptoms_encoded, diseases_encoded, test_size=0.2, random_state=42)

# Build ML model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(len(np.unique(diseases_encoded)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# Save model and encoders
model.save("health_diagnosis_model.h5")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(mlb, "symptom_encoder.pkl")

# Flask API
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['symptoms']
    symptoms_vector = mlb.transform([data])
    prediction = model.predict(symptoms_vector)
    predicted_disease = le.inverse_transform([np.argmax(prediction)])[0]
    return jsonify({"predicted_disease": predicted_disease})

if __name__ == '__main__':
    app.run(debug=True)
