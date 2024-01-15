# Start from the NVIDIA CUDA base image
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

# Avoid prompts from apt.
ENV DEBIAN_FRONTEND=noninteractive

# Preconfigure tzdata
RUN echo 'tzdata tzdata/Areas select Asia' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/Asia select Shanghai' | debconf-set-selections

# Install Python 3.8, pip, tzdata, ffmpeg in one step and clean up
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    tzdata \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip
RUN python3.8 -m pip install --upgrade pip

# Install PyTorch with GPU support
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu111

# Set the working directory in the container
WORKDIR /usr/src/app

COPY ./app/requirements-docker-gpu-lock.txt .

# Install any needed packages specified in requirements-docker-gpu.txt
RUN pip install --no-cache-dir -r requirements-docker-gpu-lock.txt

# Set the timezone
ENV TZ=Asia/Shanghai

# Copy the rest of the current directory contents into the container
COPY ./app .

# CMD to run your main application script
ENTRYPOINT ["python", "/usr/src/app/app.py"]
