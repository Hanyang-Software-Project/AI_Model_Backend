# Use the official AWS Lambda Python runtime base image
FROM public.ecr.aws/lambda/python:3.8

# Set environment variables
ENV NUMBA_CACHE_DIR=/tmp

# Install required dependencies with no cache
RUN pip install --no-cache-dir --upgrade pip \
    && pip install torch==1.9.0+cpu torchvision==0.10.0+cpu \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html \
    && pip install numpy librosa==0.9.2 numba==0.55.1 pydub boto3

# Copy the application code and model into the container
COPY app.py /var/task/app.py
COPY model.py /var/task/model.py
COPY CNN_RegDrop.pt /var/task/CNN_RegDrop.pt

# Set the working directory to /var/task
WORKDIR /var/task

# Set the PYTHONPATH to include /var/task for module imports
ENV PYTHONPATH=/var/task

# Set the Lambda handler for AWS
CMD ["app.lambda_handler"]
