#!/bin/bash
# Think More: 运行前 10 个 SWE-bench Lite 实例
# 使用方法: ./run_10_instances.sh [mode] [workers]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

MODE="${1:-run_full}"
WORKERS="${2:-2}"
TIMEOUT="${3:-1200}"

echo "=========================================="
echo "Think More: SWE-bench 实验"
echo "=========================================="
echo "模式: $MODE"
echo "并发数: $WORKERS"
echo "超时: ${TIMEOUT}s"
echo ""

# 1. 构建所有需要的 Think More 镜像
echo "步骤 1: 构建 Think More 镜像..."
cd "$PROJECT_ROOT"
python docker/build_agent_images.py --count 10

# 2. 运行批量实验
echo ""
echo "步骤 2: 运行批量实验..."
cd "$PROJECT_ROOT/experiments"
python batch_runner.py instances_10.txt "$MODE" 2 "$WORKERS" claude_code_think_more "$TIMEOUT"

# 3. 验证输出
echo ""
echo "步骤 3: 验证输出..."
TRACE_COUNT=$(find "$PROJECT_ROOT/output" -name "trace.jsonl" 2>/dev/null | wc -l)
PATCH_COUNT=$(find "$PROJECT_ROOT/output" -name "patch.diff" 2>/dev/null | wc -l)

echo "=========================================="
echo "实验完成!"
echo "=========================================="
echo "Trace 文件数: $TRACE_COUNT"
echo "Patch 文件数: $PATCH_COUNT"
echo ""
echo "输出目录: $PROJECT_ROOT/output/swebenchlite/claude_code_think_more/$MODE/"
