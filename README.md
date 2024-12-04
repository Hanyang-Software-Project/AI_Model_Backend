# AWS Backend Model for Audio Processing and Prediction

This repository contains the backend system for preprocessing audio files and generating predictions using a trained ML model on AWS. It uses a job-based approach triggered by S3 bucket events.

## Overview

- Monitors S3 for new `.wav` or `.ogg` files in the `unprocessed` folder.
- Ensures audio files are 5 seconds long and extracts Mel spectrograms.
- Runs inference using a pre-trained PyTorch model (`CNN_RegDrop`).
- Saves predictions and processed data as JSON files in the `processed` folder.
- Tags audio files in S3 as `Processed` to avoid duplication.

## How It Works

1. **Trigger**:  
   New audio files uploaded to the `unprocessed` folder in S3 trigger the Lambda function.

2. **Lambda Process**:  
   - Preprocesses files (adjusts duration, converts format).
   - Extracts Mel spectrograms and performs ML inference.
   - Saves results and processed data in the `processed` folder.

3. **File Tagging**:  
   Files are tagged as `Processed` in S3 after successful handling.

## Deployment Steps

### Prerequisites

- **AWS Account** with access to Lambda, S3, and ECR.
- **Docker Installed** for building and pushing the deployment image.

### Steps

1. **Build and Push Docker Image**:
   ```bash
   docker build -t aws-backend-model .
   docker tag aws-backend-model:latest <aws-account-id>.dkr.ecr.<region>.amazonaws.com/<repository-name>:latest
   docker push <aws-account-id>.dkr.ecr.<region>.amazonaws.com/<repository-name>:latest

2. **Set Up Lambda**:

Create a Lambda function and point it to the Docker image.
Configure S3 event triggers for the unprocessed folder.
Assign appropriate IAM permissions for S3 access.
Usage:

Upload .wav or .ogg files to the unprocessed folder in S3.
Processed results (including predictions) are saved in the processed folder as JSON.
