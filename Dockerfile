# Usee an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy only the requirements file into the container
# for better docker caching
COPY ./app/requirements-lock.txt .

# Install any needed packages specified in requirements-lock.txt
RUN pip install --no-cache-dir -r requirements-lock.txt

# Run download_resource.py during the image build
# Check if model_dev_cache exists and move it to /root/.cache if it does
COPY ./model_cache /root/.cache

# Copy the rest of the current directory contents into the container
# Copy app code at last because this is the most likely to be changed
COPY ./app .

# this will download all LLM models to the image if not exist
# this will ensure all the functionality works
RUN python ./app.py ./samples/samples --output_path ./samples/test

# CMD to run your main application script
# CMD ["python", "/usr/src/app/app.py"]
ENTRYPOINT ["python", "/usr/src/app/app.py"]
