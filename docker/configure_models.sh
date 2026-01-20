#!/bin/bash
# 配置 Claude Code + Think More 的脚本
# 在 Docker 容器启动时运行

set -e

echo "=========================================="
echo "配置 Claude Code + Think More"
echo "=========================================="

# 配置 Claude Code 模型
CLAUDE_MODEL="${CLAUDE_MODEL:-sonnet}"
CLAUDE_SETTINGS="$HOME/.claude/settings.json"

echo "配置 Claude Code 模型: $CLAUDE_MODEL"

# 如果 settings.json 不存在，创建基础配置（包含 Think More hooks）
if [ ! -f "$CLAUDE_SETTINGS" ]; then
    mkdir -p "$HOME/.claude"
    cat > "$CLAUDE_SETTINGS" <<EOF
{
  "env": {
    "ANTHROPIC_AUTH_TOKEN": "${ANTHROPIC_API_KEY}",
    "ANTHROPIC_BASE_URL": "${ANTHROPIC_BASE_URL:-https://api.anthropic.com}",
    "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "API_TIMEOUT_MS": "600000",
    "THINK_MORE_ENABLED": "1"
  },
  "permissions": {
    "allow": [],
    "deny": []
  },
  "language": "English",
  "model": "$CLAUDE_MODEL",
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "python3 /root/.claude/hooks/pytest_gate.py"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "python3 /root/.claude/hooks/post_test.py"
          }
        ]
      }
    ]
  }
}
EOF
else
    # 使用 Python 更新现有配置
    python3 << PYEOF
import json

with open("$CLAUDE_SETTINGS", "r") as f:
    config = json.load(f)

config["model"] = "$CLAUDE_MODEL"
config["language"] = "English"

if "env" not in config:
    config["env"] = {}

config["env"]["ANTHROPIC_BASE_URL"] = "${ANTHROPIC_BASE_URL:-https://api.anthropic.com}"
config["env"]["ANTHROPIC_API_KEY"] = "${ANTHROPIC_API_KEY}"
config["env"]["ANTHROPIC_AUTH_TOKEN"] = "${ANTHROPIC_API_KEY}"
config["env"]["THINK_MORE_ENABLED"] = "1"

with open("$CLAUDE_SETTINGS", "w") as f:
    json.dump(config, f, indent=2)
PYEOF
fi

echo "✓ Claude Code + Think More 配置完成: $CLAUDE_SETTINGS"

echo "=========================================="
echo "配置完成！"
echo "=========================================="
echo "Claude Code 模型: $CLAUDE_MODEL"
echo "Think More: 已启用"
echo ""
