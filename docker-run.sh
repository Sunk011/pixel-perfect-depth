#!/bin/bash

# Build the Docker image
echo "Building Docker image..."
docker build -t pixel-perfect-depth .

# Run the container
echo "Starting container..."
echo "Please open http://localhost:7860 in your browser once the server starts."

# Check if checkpoints directory exists
if [ ! -d "$(pwd)/checkpoints" ]; then
    echo "Warning: 'checkpoints' directory not found in current path. Model loading might fail."
fi

docker run -it --rm \
    --gpus all \
    -p 7860:7860 \
    -v "$(pwd)/checkpoints:/app/checkpoints" \
    -v "$(pwd)/assets:/app/assets" \
    pixel-perfect-depth
