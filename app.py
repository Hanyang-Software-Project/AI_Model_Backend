import boto3
import json
import numpy as np
from pydub import AudioSegment
import librosa
import torch
from model import CNN_RegDrop
import os

# Environment setting to avoid OpenMP conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initialize the S3 client
s3 = boto3.client('s3')

# S3 bucket name and folder paths
BUCKET_NAME = "audio-files-hanyang"
UNPROCESSED_FOLDER = "wav/"
PROCESSED_FOLDER = "mel-spectro/"
TEMP_FOLDER = "/tmp/audio_processing"

if not os.path.exists(TEMP_FOLDER):
    os.makedirs(TEMP_FOLDER)

# Load the trained model
MODEL_PATH = "CNN_RegDrop.pt"
device = "cpu"  # Change to "cuda" if running on GPU
model = CNN_RegDrop()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()


def ensure_five_seconds(audio):
    duration = 5000
    if len(audio) < duration:
        silence = AudioSegment.silent(duration=duration - len(audio))
        audio = audio + silence
    elif len(audio) > duration:
        audio = audio[:duration]
    return audio


def create_mel_spectrogram(file_path, target_width=431):
    y, sr = librosa.load(file_path, sr=22050)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    if mel_spec_db.shape[1] < target_width:
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, target_width - mel_spec_db.shape[1])), mode='constant')
    elif mel_spec_db.shape[1] > target_width:
        mel_spec_db = mel_spec_db[:, :target_width]
    return mel_spec_db.reshape(1, 1, 128, target_width).astype(np.float32)


def predict_with_model(data):
    data_tensor = torch.tensor(data).to(device)
    with torch.no_grad():
        outputs = model(data_tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
    pred_class = np.argmax(probabilities)
    pred_probability = probabilities[0][pred_class]
    anomaly_score = -np.log(pred_probability)
    return {
        "predicted_class": int(pred_class),
        "predicted_probability": float(pred_probability),
        "anomaly_score": float(anomaly_score)
    }


def tag_file_as_processed(bucket_name, key):
    s3.put_object_tagging(
        Bucket=bucket_name,
        Key=key,
        Tagging={
            "TagSet": [
                {"Key": "Status", "Value": "Processed"}
            ]
        }
    )
    print(f"Tagged {key} as processed")


def process_wav_file(bucket_name, object_key):
    print(f"Processing file: {object_key}")
    local_path = os.path.join(TEMP_FOLDER, os.path.basename(object_key))
    s3.download_file(bucket_name, object_key, local_path)
    audio = AudioSegment.from_wav(local_path)
    adjusted_audio = ensure_five_seconds(audio)
    adjusted_file_path = os.path.join(TEMP_FOLDER, "adjusted_" + os.path.basename(object_key))
    adjusted_audio.export(adjusted_file_path, format="wav")
    mel_spec = create_mel_spectrogram(adjusted_file_path)
    prediction = predict_with_model(mel_spec)
    print(f"Inference result for {object_key}: {prediction}")
    payload = {"data": mel_spec.tolist(), "prediction": prediction}
    json_file_path = os.path.basename(object_key).replace(".wav", ".json")
    processed_s3_key = f"{PROCESSED_FOLDER}{json_file_path}"
    s3.put_object(
        Bucket=bucket_name,
        Key=processed_s3_key,
        Body=json.dumps(payload),
        ContentType="application/json"
    )
    print(f"Saved JSON for {object_key} to {processed_s3_key}")
    tag_file_as_processed(bucket_name, object_key)


def lambda_handler(event, context):
    for record in event['Records']:
        bucket_name = record['s3']['bucket']['name']
        object_key = record['s3']['object']['key']
        if object_key.endswith(".wav") and object_key.startswith(UNPROCESSED_FOLDER):
            process_wav_file(bucket_name, object_key)
