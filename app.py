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
@app.route('/', methods=['GET'])
def home():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
      <title>Threat Detection</title>
      <style>
        body { font-family: Arial; margin: 20px; padding: 20px; }
        label, input { display: block; margin-top: 10px; }
        button { margin-top: 20px; }
      </style>
    </head>
    <body>
      <h2>Threat Detection Predictor</h2>
      <form id="predictionForm">
        <label>srv_count: <input type="number" name="srv_count" required></label>
        <label>service: <input type="text" name="service" required></label>
        <label>src_bytes: <input type="number" name="src_bytes" required></label>
        <label>count: <input type="number" name="count" required></label>
        <label>protocol_type: <input type="text" name="protocol_type" required></label>
        <label>dst_host_same_src_port_rate: <input type="number" step="any" name="dst_host_same_src_port_rate" required></label>
        <label>dst_host_diff_srv_rate: <input type="number" step="any" name="dst_host_diff_srv_rate" required></label>
        <label>diff_srv_rate: <input type="number" step="any" name="diff_srv_rate" required></label>
        <label>dst_bytes: <input type="number" name="dst_bytes" required></label>
        <label>dst_host_same_srv_rate: <input type="number" step="any" name="dst_host_same_srv_rate" required></label>
        <button type="submit">Predict</button>
      </form>

      <h3 id="result"></h3>

      <script>
        document.getElementById("predictionForm").addEventListener("submit", async function(e) {
          e.preventDefault();
          const formData = new FormData(this);
          const jsonData = {};
          formData.forEach((value, key) => {
            jsonData[key] = isNaN(value) ? value : Number(value);
          });

          const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(jsonData)
          });

          const result = await response.json();
          document.getElementById("result").innerText = 
            result.prediction !== undefined ? `Prediction: ${result.prediction}` : `Error: ${result.error}`;
        });
      </script>
    </body>
    </html>
    '''


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
