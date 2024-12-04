import json
import boto3
import os
os.environ["NUMBA_CACHE_DIR"] = "/tmp"
from pydub import AudioSegment
import librosa
import torch
import numpy as np
from model import CNN_RegDrop

s3 = boto3.client('s3')

# Load the trained model
MODEL_PATH = "CNN_RegDrop.pt"
device = "cpu"
model = CNN_RegDrop()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()


def lambda_handler(event, context):
    try:
        for record in event['Records']:
            bucket_name = record['s3']['bucket']['name']
            object_key = record['s3']['object']['key']
            print(f"Processing file: {object_key} from bucket: {bucket_name}")

            # Download the file locally
            temp_file_path = f"/tmp/{os.path.basename(object_key)}"
            s3.download_file(bucket_name, object_key, temp_file_path)

            # Process the file (adjust duration, create Mel spectrogram, infer with model)
            audio = AudioSegment.from_wav(temp_file_path)
            adjusted_audio = ensure_five_seconds(audio)

            # Save the adjusted audio locally
            adjusted_file_path = f"/tmp/adjusted_{os.path.basename(object_key)}"
            adjusted_audio.export(adjusted_file_path, format="wav")

            # Create Mel spectrogram
            mel_spec = create_mel_spectrogram(adjusted_file_path, target_width=431)

            # Perform inference
            prediction = predict_with_model(mel_spec)
            print(f"Prediction: {prediction}")

            # Save prediction to S3 in the processed folder
            json_payload = json.dumps({
                "prediction": prediction,
                "source_file": object_key
            })

            s3.put_object(
                Bucket=bucket_name,
                Key=f"processed/{os.path.basename(object_key)}.json",
                Body=json_payload,
                ContentType="application/json"
            )

            print(f"Processed file {object_key} and saved results to processed/")

    except Exception as e:
        print(f"Error processing file: {e}")
        raise


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

    # Pad or trim the Mel spectrogram to match the target dimensions
    if mel_spec_db.shape[1] < target_width:
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, target_width - mel_spec_db.shape[1])), mode='constant')
    elif mel_spec_db.shape[1] > target_width:
        mel_spec_db = mel_spec_db[:, :target_width]

    # Reshape to the required dimensions
    mel_spec_reshaped = mel_spec_db.reshape(1, 1, 128, target_width).astype(np.float32)
    return mel_spec_reshaped


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
