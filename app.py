from flask import Flask, request, jsonify
import joblib
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load model and encoders
model = tf.keras.models.load_model("cnn_lstm_model.h5")
label_encoders = joblib.load("label_encoder.pkl")

features = ['srv_count', 'service', 'src_bytes', 'count', 'protocol_type',
            'dst_host_same_src_port_rate', 'dst_host_diff_srv_rate', 'diff_srv_rate',
            'dst_bytes', 'dst_host_same_srv_rate']

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = [data[feature] for feature in features]
    input_array = np.array([input_data]).reshape(1, len(features), 1)  # reshape for CNN-LSTM
    prediction = model.predict(input_array)
    result = prediction.argmax(axis=1)[0]
    return jsonify({"prediction": int(result)})

@app.route('/', methods=['GET'])
def home():
    return "Threat Detection Model is Live!"
