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

    # Input validation
    required_fields = set(features)
    if not required_fields.issubset(data.keys()):
        return jsonify({"error": f"Missing input fields. Required: {list(required_fields)}"}), 400

    # Encode categorical fields using label encoders
    for col in ['service', 'protocol_type']:
        le = label_encoders[col]
        if data[col] not in le.classes_:
            return jsonify({"error": f"Invalid value for {col}: '{data[col]}'"}), 400
        data[col] = le.transform([data[col]])[0]

    try:
        # Prepare input for model
        input_data = [data[feature] for feature in features]
        input_array = np.array([input_data]).reshape(1, len(features), 1)
        
        # Prediction
        prediction = model.predict(input_array)
        result = int(prediction.argmax(axis=1)[0])
        
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
