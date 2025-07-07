#!/bin/bash

# Navigate to the directory containing the script
cd "$(dirname "$0")" || exit 1

# Check for GPU availability (NVIDIA GPU via nvidia-smi)
echo "Checking for GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "GPU is available. Using GPU configuration."
    COMPOSE_FILE="./docker/docker-compose.gpu.yml"
else
    echo "GPU is not available. Using CPU configuration."
    COMPOSE_FILE="./docker/docker-compose.cpu.yml"
fi

# Start docker-compose with the appropriate file
echo "Starting docker compose..."
docker compose -f "$COMPOSE_FILE" up -d

# Set the predefined token (must match the token in docker-compose.yml)
TOKEN="351"

# Construct the Jupyter Notebook URL
JUPYTER_URL="http://127.0.0.1:8888?token=${TOKEN}"

# Open the URL in the default browser
echo "Opening Jupyter Notebook in the default browser..."
open "$JUPYTER_URL"

# Wait for the user to manually close the browser
# Since macOS doesn't provide a simple way to track the browser process, we prompt the user
read -rp "Press ENTER after closing the Jupyter Notebook in your browser..."

# Stop and remove the container
echo "Stopping and removing the container..."
docker compose -f "$COMPOSE_FILE" down
echo "Jupyter Notebook closed and container stopped."
