#!/bin/bash
# Run Think More SWE-bench evaluation
# Usage: ./run_evaluation.sh [mode] [dataset] [max_instances]
# Example: ./run_evaluation.sh full lite 10

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

MODE="${1:-full}"          # baseline, skills_only, skills_hooks, full
DATASET="${2:-lite}"       # lite, verified
MAX_INSTANCES="${3:-10}"

echo "🚀 Running Think More SWE-bench Evaluation"
echo "   Mode: $MODE"
echo "   Dataset: $DATASET"
echo "   Max instances: $MAX_INSTANCES"
echo ""

# Check for API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "❌ Error: ANTHROPIC_API_KEY not set"
    echo "   Export it: export ANTHROPIC_API_KEY=your-key"
    exit 1
fi

# Select container based on mode
if [ "$MODE" == "baseline" ]; then
    CONTAINER="baseline-runner"
else
    CONTAINER="think-more-runner"
fi

# Start container if not running
if ! docker ps | grep -q "$CONTAINER"; then
    echo "📦 Starting $CONTAINER..."
    cd "$SCRIPT_DIR"
    docker-compose up -d "$CONTAINER"
fi

# Run experiment
echo "🔬 Running experiment..."
docker exec -it "$CONTAINER" python /workspace/experiments/run_experiment.py \
    /workspace/experiments/configs/${DATASET}_${MODE}.yaml

echo ""
echo "✅ Evaluation complete!"
echo "📁 Results saved to: $PROJECT_ROOT/experiments/output/"
