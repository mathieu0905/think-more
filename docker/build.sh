#!/bin/bash
# Build Think More Docker images for SWE-bench evaluation
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🔧 Building Think More Docker images..."

# Default base image
BASE_IMAGE="${BASE_IMAGE:-swebench/sweb.eval.x86_64.base:latest}"

echo "📦 Base image: $BASE_IMAGE"

# Build the Think More overlay image
docker build \
    --build-arg BASE_IMAGE="$BASE_IMAGE" \
    -t think-more:latest \
    -f "$SCRIPT_DIR/Dockerfile.think-more" \
    "$PROJECT_ROOT"

echo "✅ Build complete!"
echo ""
echo "Usage:"
echo "  # Start interactive shell:"
echo "  docker run -it --rm -v \$(pwd):/testbed think-more:latest bash"
echo ""
echo "  # Or use docker-compose:"
echo "  cd $SCRIPT_DIR && docker-compose up -d think-more-runner"
echo "  docker exec -it think-more-runner bash"
