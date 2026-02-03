#!/bin/bash

# Container name
CONTAINER_NAME="bge-m3-sparse-cpu"

# Tag (default: latest)
TAG="${1:-latest}"

# Stop existing container
if [ "$(docker ps -aq -f name=^/${CONTAINER_NAME}$)" ]; then
    echo "Stopping existing container..."
    docker stop ${CONTAINER_NAME} && docker rm ${CONTAINER_NAME}
fi

# Start container with optimized settings for CPU
echo "Starting ${CONTAINER_NAME}:${TAG}..."
docker run -d --name ${CONTAINER_NAME} \
  -p 8080:8080 \
  --cpus=32 \
  --memory=16g \
  -e HF_HUB_OFFLINE=1 \
  -e WORKERS=1 \
  -e OMP_NUM_THREADS=32 \
  -e MKL_NUM_THREADS=32 \
  -e MICRO_BATCH_SIZE=4 \
  -e MAX_LENGTH=512 \
  bge-m3-sparse-cpu:${TAG}

echo ""
docker ps -f name=${CONTAINER_NAME}
