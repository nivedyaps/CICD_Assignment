# Use an official Python runtime as a base image
FROM python:3.8-slim
 
# Set the working directory inside the container
WORKDIR /app
 
# Copy the current directory contents into the container at /app
COPY . /app
 
# Install any dependencies required for training and testing
RUN pip install -r requirements.txt
 
# Train the model during the build phase
RUN python train.py
 
# Command to execute test.py when the container is run
CMD ["python", "test.py"]