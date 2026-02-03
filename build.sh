#!/bin/bash
# Build Docker images for BGE-M3 Sparse Service

set -e

TAG="${TAG:-latest}"
PROXY_ARGS=""

# Add proxy if available
if [ -n "$http_proxy" ]; then
    PROXY_ARGS="--build-arg http_proxy=$http_proxy --build-arg https_proxy=$https_proxy"
fi

echo "============================================"
echo "Building BGE-M3 Sparse Service Images"
echo "Tag: $TAG"
echo "============================================"

echo ""
echo "=== Building GPU image (with model weights) ==="
DOCKER_BUILDKIT=1 docker build $PROXY_ARGS -t bge-m3-sparse:$TAG -f Dockerfile .
echo "✅ GPU image built: bge-m3-sparse:$TAG"

echo ""
echo "=== Building CPU image (with model weights) ==="
DOCKER_BUILDKIT=1 docker build $PROXY_ARGS -t bge-m3-sparse-cpu:$TAG -f Dockerfile-cpu .
echo "✅ CPU image built: bge-m3-sparse-cpu:$TAG"

echo ""
echo "============================================"
echo "Build Complete!"
echo "============================================"
docker images | grep bge-m3-sparse
echo ""
echo "Run GPU: docker run --gpus all -p 8080:8080 bge-m3-sparse:$TAG"
echo "Run CPU: docker run -p 8080:8080 bge-m3-sparse-cpu:$TAG"
