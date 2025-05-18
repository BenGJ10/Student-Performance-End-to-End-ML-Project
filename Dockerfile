# Use official Python image as base
FROM python:3.9-slim

# Set working directory in container
WORKDIR /app

# Copy project files into container
COPY . /app

# Install dependencies
RUN pip install -r requirements.txt

# Specifies the default command to run when the container starts    
CMD ["python", "app.py"]