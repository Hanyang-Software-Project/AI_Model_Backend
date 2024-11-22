import numpy as np
import torch
from flask import Flask, request, jsonify
from apig_wsgi import make_lambda_handler
from src.model import CNN_RegDrop

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "model/CNN_RegDrop.pt"
device = "cpu"  # Use CPU for inference
model = CNN_RegDrop()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()  # Set model to evaluation mode


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse JSON input
        input_data = request.json

        # Extract and preprocess input data
        data = np.array(input_data["data"]).reshape(1, 1, 128, 431).astype(np.float32)
        data_tensor = torch.tensor(data).to(device)

        # Run the model
        with torch.no_grad():
            outputs = model(data_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()

        # Interpret the results
        pred_class = np.argmax(probabilities)
        pred_probability = probabilities[0][pred_class]
        anomaly_score = -np.log(pred_probability)

        # Return the results as JSON
        return jsonify({
            "predicted_class": int(pred_class),
            "predicted_probability": float(pred_probability),
            "anomaly_score": float(anomaly_score)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/", methods=["GET"])
def home():
    return "Flask App is Running!"


# Define the AWS Lambda handler
#lambda_handler = make_lambda_handler(app)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
