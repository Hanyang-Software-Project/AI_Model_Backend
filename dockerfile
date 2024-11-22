# Use the official AWS Lambda Python runtime base image
FROM public.ecr.aws/lambda/python:3.8

# Install required dependencies with no cache
RUN pip install --no-cache-dir --upgrade pip \
    && pip install torch==1.9.0+cpu torchvision==0.10.0+cpu \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html \
    && pip install flask numpy apig-wsgi

# Copy the application code and model into the container
COPY src /var/task/src
COPY model/CNN_RegDrop.pt /var/task/model/CNN_RegDrop.pt

# Set the working directory to /var/task
WORKDIR /var/task

# Set the PYTHONPATH to include /var/task for module imports
ENV PYTHONPATH=/var/task

# Set the Lambda handler for AWS
CMD ["src.app.lambda_handler"]
