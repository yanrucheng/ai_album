# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements-lock.txt

# Run download_resource.py during the image build
RUN python ./download_resource.py

# CMD to run your main application script
CMD ["python", "./app.py"]
