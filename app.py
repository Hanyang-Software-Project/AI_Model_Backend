import numpy as np
import torch
from flask import Flask, request, jsonify
from apig_wsgi import make_lambda_handler
from model import CNN_RegDrop
import uuid
import boto3
import json
import logging

app = Flask(__name__)

s3 = boto3.client('s3')
BUCKET_NAME = 'audio-json'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the trained model
MODEL_PATH = "CNN_RegDrop.pt"
device = "cpu"
model = CNN_RegDrop()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()


@app.route("/predict", methods=["POST"])
def predict():
    try:
        logger.info("Received request.")
        input_data = request.json
        logger.info(f"Input data: {input_data}")

        # Save input data to S3
        unique_id = str(uuid.uuid4())
        s3_key = f"input-data/{unique_id}.json"
        logger.info(f"Generated S3 key: {s3_key}")

        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=s3_key,
            Body=json.dumps(input_data),
            ContentType="application/json"
        )
        logger.info(f"Saved input data to S3: {s3_key}")

        # Process data for prediction
        data = np.array(input_data["data"]).reshape(1, 1, 128, 431).astype(np.float32)
        data_tensor = torch.tensor(data).to(device)
        logger.info("Data tensor created.")

        with torch.no_grad():
            outputs = model(data_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()

        pred_class = np.argmax(probabilities)
        pred_probability = probabilities[0][pred_class]
        anomaly_score = -np.log(pred_probability)

        logger.info(f"Prediction: {pred_class}, Probability: {pred_probability}, Anomaly Score: {anomaly_score}")

        return jsonify({
            "predicted_class": int(pred_class),
            "predicted_probability": float(pred_probability),
            "anomaly_score": float(anomaly_score),
            "s3_key": s3_key
        })
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 400


# Define the AWS Lambda handler
lambda_handler = make_lambda_handler(app)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
