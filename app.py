import numpy as np
import torch
import boto3
import json
import logging
import time
import requests
from model import CNN_RegDrop

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

s3 = boto3.client('s3')
BUCKET_NAME = 'audio-json'
INPUT_FOLDER = 'input-data/'
PROCESSED_FOLDER = 'processed-data/'

NOTIFICATION_API_URL = 'https://example.com/notify'

MODEL_PATH = "CNN_RegDrop.pt"
device = "cpu"
model = CNN_RegDrop()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()


def fetch_s3_keys(bucket_name, prefix):
    """
    Fetches the list of keys (file paths) from an S3 bucket folder.
    """
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    if "Contents" in response:
        keys = [obj["Key"] for obj in response["Contents"] if obj["Key"] != prefix]
        return keys
    return []


def process_file_from_s3(bucket_name, key):
    """
    Fetches a file from S3, processes it, and sends a notification if anomaly is detected.
    """
    response = s3.get_object(Bucket=bucket_name, Key=key)
    input_data = json.loads(response["Body"].read().decode("utf-8"))

    data = np.array(input_data["data"]).reshape(1, 1, 128, 431).astype(np.float32)
    data_tensor = torch.tensor(data).to(device)

    with torch.no_grad():
        outputs = model(data_tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()

    pred_class = np.argmax(probabilities)
    pred_probability = probabilities[0][pred_class]
    anomaly_score = -np.log(pred_probability)

    logger.info(f"Prediction: {pred_class}, Probability: {pred_probability}, Anomaly Score: {anomaly_score}")

    if pred_class == 1:
        notification_payload = {
            "s3_key": key,
            "anomaly_score": anomaly_score,
            "predicted_class": int(pred_class),
            "predicted_probability": float(pred_probability)
        }
        requests.post(NOTIFICATION_API_URL, json=notification_payload)

    new_key = key.replace(INPUT_FOLDER, PROCESSED_FOLDER)
    s3.copy_object(Bucket=bucket_name, CopySource={"Bucket": bucket_name, "Key": key}, Key=new_key)
    s3.delete_object(Bucket=bucket_name, Key=key)


def monitor_s3_bucket():
    """
    Continuously monitors the S3 bucket for new data and processes it.
    """
    while True:
        keys = fetch_s3_keys(BUCKET_NAME, INPUT_FOLDER)
        for key in keys:
            process_file_from_s3(BUCKET_NAME, key)
        time.sleep(60)


if __name__ == "__main__":
    monitor_s3_bucket()
