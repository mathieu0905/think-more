#!/bin/bash
# Build Think More overlay for multiple SWE-bench repo images
# Usage: ./build_repo_images.sh [repo1] [repo2] ...
# Example: ./build_repo_images.sh django sympy astropy

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default repos to build
REPOS="${@:-django sympy astropy matplotlib}"

echo "🔧 Building Think More images for repos: $REPOS"

for repo in $REPOS; do
    BASE_IMAGE="swebench/sweb.eval.x86_64.${repo}:latest"
    TARGET_IMAGE="think-more-${repo}:latest"

    echo ""
    echo "📦 Building $TARGET_IMAGE from $BASE_IMAGE..."

    # Check if base image exists
    if ! docker image inspect "$BASE_IMAGE" &>/dev/null; then
        echo "⚠️  Base image $BASE_IMAGE not found, pulling..."
        docker pull "$BASE_IMAGE" || {
            echo "❌ Failed to pull $BASE_IMAGE, skipping..."
            continue
        }
    fi

    docker build \
        --build-arg BASE_IMAGE="$BASE_IMAGE" \
        -t "$TARGET_IMAGE" \
        -f "$SCRIPT_DIR/Dockerfile.think-more" \
        "$PROJECT_ROOT"

    echo "✅ Built $TARGET_IMAGE"
done

echo ""
echo "🎉 All images built successfully!"
echo ""
echo "Available images:"
docker images | grep think-more
