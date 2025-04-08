# Usee an official Python runtime as a parent image
FROM --platform=linux/amd64 python:3.8-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Update package lists and install FFmpeg
RUN apt-get update \
    && apt-get install -y ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file into the container
# for better docker caching
COPY ./app/requirements-docker-lock.txt .

# Install any needed packages specified in requirements-lock.txt
RUN pip install --no-cache-dir -r requirements-docker-lock.txt

# Copy the rest of the current directory contents into the container
# Copy app code at last because this is the most likely to be changed
COPY ./app .

# CMD to run your main application script
# CMD ["python", "/usr/src/app/app.py"]
ENTRYPOINT ["python", "/usr/src/app/app.py"]
