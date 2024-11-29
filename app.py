import os
import boto3
import json
import numpy as np
from pydub import AudioSegment
import librosa

# Initialize the S3 client
s3 = boto3.client('s3')

# S3 bucket name
BUCKET_NAME = "audio-files-hanyang"

# Paths for unprocessed and processed folders
UNPROCESSED_FOLDER = "unprocessed/"
PROCESSED_FOLDER = "processed/"

# Local temporary folder for processing
TEMP_FOLDER = "/tmp/audio_processing"
if not os.path.exists(TEMP_FOLDER):
    os.makedirs(TEMP_FOLDER)


def download_wav_files_from_s3(bucket_name, prefix):
    """
    Download .wav files from the specified S3 folder to the local temporary folder.
    """
    files = []
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    for obj in response.get("Contents", []):
        if obj["Key"].endswith(".wav"):
            local_path = os.path.join(TEMP_FOLDER, os.path.basename(obj["Key"]))
            s3.download_file(bucket_name, obj["Key"], local_path)
            files.append(local_path)
    return files


def ensure_five_seconds(audio):
    """
    Adjust the audio to be exactly 5 seconds long.
    If shorter, pad with silence; if longer, trim to 5 seconds.
    """
    duration = 5000
    if len(audio) < duration:
        silence = AudioSegment.silent(duration=duration - len(audio))
        audio = audio + silence
    elif len(audio) > duration:
        audio = audio[:duration]
    return audio


def create_mel_spectrogram(file_path):
    """
    Create a Mel spectrogram from a .wav file.
    """
    y, sr = librosa.load(file_path, sr=22050)  # Load audio file with librosa
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def process_wav_files(bucket_name):
    """
    Process .wav files from the S3 bucket:
    - Ensure 5 seconds duration
    - Create Mel spectrograms
    - Create JSON payload for each file
    - Save JSON to S3
    """
    # Download all .wav files from the "unprocessed" folder in the S3 bucket
    files = download_wav_files_from_s3(bucket_name, UNPROCESSED_FOLDER)

    for file_path in files:
        # Load the audio file with pydub
        audio = AudioSegment.from_wav(file_path)

        # Ensure the audio is exactly 5 seconds long
        adjusted_audio = ensure_five_seconds(audio)

        # Save the adjusted audio to a temporary file
        adjusted_file_path = os.path.join(TEMP_FOLDER, "adjusted_" + os.path.basename(file_path))
        adjusted_audio.export(adjusted_file_path, format="wav")

        # Create Mel spectrogram
        mel_spec = create_mel_spectrogram(adjusted_file_path)

        # Convert the Mel spectrogram to JSON payload
        data_list = mel_spec.tolist()
        payload = {
            "data": data_list
        }

        # Save the JSON payload to the "processed" folder in S3
        json_file_name = os.path.basename(file_path).replace(".wav", ".json")
        s3_key = f"{PROCESSED_FOLDER}{json_file_name}"
        s3.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=json.dumps(payload),
            ContentType="application/json"
        )

        print(f"Processed and saved JSON for {file_path} to S3: {s3_key}")


# Run the processing function
if __name__ == "__main__":
    process_wav_files(BUCKET_NAME)
