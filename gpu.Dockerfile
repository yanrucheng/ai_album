# Start from the NVIDIA CUDA base image
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

# Install Python 3.8
RUN apt-get update && apt-get install -y python3.8 python3-pip

# Upgrade pip
RUN python3.8 -m pip install --upgrade pip

# Install PyTorch with GPU support
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu111

# Set the working directory in the container
WORKDIR /usr/src/app

COPY ./app/requirements-docker-gpu.txt .

# Install any needed packages specified in requirements-lock.txt
RUN pip install --no-cache-dir -r requirements-docker-gpu.txt

# Update package lists and install FFmpeg
RUN apt-get update \
    && apt-get install -y ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy the rest of the current directory contents into the container
# Copy app code at last because this is the most likely to be changed
COPY ./app .

# CMD to run your main application script
# CMD ["python", "/usr/src/app/app.py"]
ENTRYPOINT ["python", "/usr/src/app/app.py"]
