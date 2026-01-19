# 研究方案：强制结构化推理约束对 LLM Agent 调试行为的影响

> 生成日期：2025-01-19

---

## 第一部分：研究问题与定位

### 1.1 原始问题

AI coding agent 在多步调试任务中表现出"懒惰"行为：
- 跳过深入的数据流分析
- 直接运行测试并进入"边跑边猜"模式
- 倾向于在症状位置打补丁，而非追溯根因

### 1.2 研究问题

**核心表述：**
> "强制输出结构化推理记录能改善 agent 在多步调试任务中的表现，且过程证据表明 agent 进行了更接近根因的分析"

**关键区别：**
- 不过度声称 agent "真的在深度推理"
- 用间接证据（patch 位置、证据引用率）支撑"更深入"的说法
- 机制可以是黑盒

### 1.3 研究贡献定位

**可以声称：**
1. 一种可落地的 agent 行为约束机制（Skills + Hooks + MCP）
2. 该机制在 SWE-bench 上的效果量化
3. 过程指标与结果的相关性分析
4. 过程证据表明 agent 进行了更接近根因的分析

**支撑"更深入"的间接指标：**

| 指标 | 测量方式 | 为什么能支撑 |
|------|----------|-------------|
| Patch 位置 | 补丁距离 root cause 的代码距离 | 更深入 → 更接近 def 而非 use-site |
| 假设具体性 | 人工标注 intent/prediction 的信息量 | 更深入 → 写出具体变量名、预期值 |
| 证据引用率 | state.json 中是否引用了 MCP 返回的数据流信息 | 更深入 → 真的使用了工具证据 |
| 分支收敛模式 | 假设是被证据淘汰还是被结果覆盖 | 更深入 → 有逻辑地排除假设 |

---

## 第二部分：技术架构

### 2.1 三层架构总览

```
┌─────────────────────────────────────────────────────┐
│                    Agent (Claude)                    │
└─────────────────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   Skills    │  │    Hooks    │  │     MCP     │
│  (协议层)    │  │   (控制层)   │  │   (证据层)  │
│             │  │             │  │             │
│ 定义"怎么想" │  │ 强制执行+日志│  │ 数据流工具  │
└─────────────┘  └─────────────┘  └─────────────┘
         │               │               │
         └───────────────┼───────────────┘
                         ▼
              ┌─────────────────┐
              │   state.json    │
              │   (融合锚点)     │
              └─────────────────┘
```

### 2.2 各层职责

| 层级 | 组件 | 职责 | 实现方式 |
|------|------|------|----------|
| 协议层 | Skills | 定义推理结构模板 | Claude Code Skills |
| 控制层 | Hooks | 测试前置检查 + 全程日志 | Claude Code Hooks (PreToolUse/PostToolUse) |
| 证据层 | MCP | 提供数据流分析能力 | Jedi + PyCG 封装为 MCP Server |

### 2.3 state.json 作为融合锚点

三层都围绕 `state.json` 工作：
- **Skills** 定义它的 schema
- **Hooks** 检查它、记录它
- **MCP** 填充它的证据字段

---

## 第三部分：state.json Schema 设计

```json
{
  "version": 1,
  "task_id": "django__django-12345",

  "hypotheses": [
    {
      "id": "h1",
      "description": "QuerySet.filter() 在空列表时返回 None 而非空 QuerySet",
      "status": "active | eliminated | confirmed",
      "evidence": []
    }
  ],

  "dataflow_chain": {
    "summary": "user_input → view.get_queryset() → QuerySet.filter() → None",
    "mcp_callgraph": null,
    "mcp_defuse": null
  },

  "current_probe": {
    "intent": "验证 filter([]) 的返回值类型",
    "prediction": {
      "if_pass": "h1 被排除，问题在其他位置",
      "if_fail": "h1 被确认，需要修复 filter 实现"
    },
    "test_command": "pytest tests/queryset/test_filter.py -k empty"
  },

  "history": [
    {
      "round": 1,
      "probe": { },
      "result": "fail",
      "update": "h1 确认，开始定位 filter 实现",
      "timestamp": "2024-01-19T12:00:00Z"
    }
  ]
}
```

### 关键字段说明

| 字段 | 用途 | 谁写入 | 谁检查 |
|------|------|--------|--------|
| `hypotheses` | 当前候选假设（最多2个） | Agent (Skill 约束) | Hooks (收敛检查) |
| `dataflow_chain` | 数据流分析结果 | Agent + MCP | 用于计算"证据引用率" |
| `current_probe.intent` | 本次测试目的 | Agent | Hooks (pytest gate) |
| `current_probe.prediction` | 预测结果含义 | Agent | Hooks (pytest gate) |
| `history` | 完整推理历史 | Hooks 自动追加 | 用于后续分析 |

---

## 第四部分：Hooks 实现设计

### 4.1 Hook 配置结构

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [{ "type": "command", "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/pytest_gate.py" }]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Bash",
        "hooks": [{ "type": "command", "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/post_test.py" }]
      }
    ],
    "Stop": [
      {
        "matcher": "",
        "hooks": [{ "type": "command", "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/anti_cheat.py" }]
      }
    ]
  }
}
```

### 4.2 三个核心 Hook

| Hook | 触发时机 | 职责 |
|------|----------|------|
| `pytest_gate.py` | pytest 执行前 | 检查 state.json 是否有 intent + prediction |
| `post_test.py` | pytest 执行后 | 记录结果，强制要求 update |
| `anti_cheat.py` | Agent 结束时 | 检测投机性 patch（try/except 吞错等） |

### 4.3 pytest_gate.py 核心逻辑

```python
#!/usr/bin/env python3
import json, sys, os, re

input_data = json.load(sys.stdin)
command = input_data.get("tool_input", {}).get("command", "")

# 检测 pytest 及变体
PYTEST_PATTERNS = [r"\bpytest\b", r"\bpy\.test\b", r"python\s+-m\s+pytest"]
is_pytest = any(re.search(p, command) for p in PYTEST_PATTERNS)

if not is_pytest:
    sys.exit(0)

# 检查 state.json
state_path = os.path.join(input_data["cwd"], "state.json")

if not os.path.exists(state_path):
    print("❌ 运行测试前必须创建 state.json", file=sys.stderr)
    sys.exit(2)

with open(state_path) as f:
    state = json.load(f)

probe = state.get("current_probe", {})
if not probe.get("intent") or not probe.get("prediction"):
    print("❌ state.json 缺少 intent 或 prediction", file=sys.stderr)
    print("请先说明：这次测试要验证什么假设？", file=sys.stderr)
    sys.exit(2)

# 检查是否有未回填的历史
history = state.get("history", [])
if history and not history[-1].get("update"):
    print("❌ 上一轮测试结果未回填 update", file=sys.stderr)
    sys.exit(2)

sys.exit(0)
```

### 4.4 post_test.py 核心逻辑

```python
#!/usr/bin/env python3
import json, sys, os
from datetime import datetime

input_data = json.load(sys.stdin)
command = input_data.get("tool_input", {}).get("command", "")
response = input_data.get("tool_response", {})

# 只处理 pytest
if "pytest" not in command:
    sys.exit(0)

# 记录到 trace.jsonl
trace_path = os.path.join(input_data["cwd"], "trace.jsonl")
trace_entry = {
    "timestamp": datetime.now().isoformat(),
    "event": "test_executed",
    "command": command,
    "exit_code": response.get("exitCode"),
    "output_length": len(response.get("stdout", "")),
}

with open(trace_path, "a") as f:
    f.write(json.dumps(trace_entry) + "\n")

# 提醒 agent 更新 state
output = {
    "hookSpecificOutput": {
        "hookEventName": "PostToolUse",
        "additionalContext": "请立即更新 state.json：1) 记录 result 2) 填写 update 3) 更新 hypotheses 状态"
    }
}
print(json.dumps(output))
sys.exit(0)
```

---

## 第五部分：MCP 工具设计

### 5.1 技术选型

| 组件 | 用途 | 选择理由 |
|------|------|----------|
| **Jedi** | goto_definition, find_references | 成熟、纯 Python、LSP 兼容 |
| **PyCG** | 调用图生成 | 轻量、学术项目、专注 call graph |

### 5.2 MCP Server 接口设计

```python
# 两个核心工具

@mcp.tool()
def trace_symbol(symbol: str, file_path: str = None) -> dict:
    """
    追踪符号的定义和引用位置

    返回：
    {
        "definitions": [{"file": "...", "line": 10, "context": "def foo():"}],
        "references": [{"file": "...", "line": 25, "context": "result = foo()"}],
        "summary": "foo 定义于 module.py:10，被 3 处调用"
    }
    """

@mcp.tool()
def trace_callchain(entry_point: str, direction: str = "callers") -> dict:
    """
    追踪函数的调用链

    direction: "callers" (谁调用了它) | "callees" (它调用了谁)

    返回：
    {
        "chain": [
            {"func": "main", "file": "app.py", "line": 5},
            {"func": "process", "file": "core.py", "line": 20},
            {"func": "validate", "file": "utils.py", "line": 45}
        ],
        "summary": "main → process → validate (3 层调用)"
    }
    """
```

### 5.3 实现架构

```
┌─────────────────────────────────────────┐
│            MCP Server (Python)          │
├─────────────────────────────────────────┤
│  trace_symbol()      trace_callchain()  │
├─────────────────────────────────────────┤
│         Adapter Layer                   │
│   ┌─────────────┐  ┌─────────────┐      │
│   │    Jedi     │  │    PyCG     │      │
│   │ definitions │  │ call graph  │      │
│   │ references  │  │             │      │
│   └─────────────┘  └─────────────┘      │
└─────────────────────────────────────────┘
```

### 5.4 与 state.json 的集成

Agent 调用 MCP 工具后，应将结果写入 state.json：

```json
{
  "dataflow_chain": {
    "summary": "user_input → validate() → process() → 返回 None",
    "mcp_callgraph": {
      "tool": "trace_callchain",
      "input": {"entry_point": "process", "direction": "callers"},
      "output": { }
    },
    "mcp_defuse": {
      "tool": "trace_symbol",
      "input": {"symbol": "user_input"},
      "output": { }
    }
  }
}
```

---

## 第六部分：实验设计

### 6.1 数据集策略

| 阶段 | 数据集 | 用途 | 样本量 |
|------|--------|------|--------|
| 开发调参 | SWE-bench Lite | 快速迭代机制 | 全量 (~300) |
| 强验收 | SWE-bench Verified (子集) | 可信结论 | 抽样 20-50 |

**语言范围：Python-only**
- SWE-bench 本身是 Python
- 论文明确 scope，多语言作为 future work

### 6.2 实验对照组

| 组别 | 配置 | 验证目标 |
|------|------|----------|
| **G0** | Baseline (原生 Claude Code) | 基准线 |
| **G1** | + Skills only | "自觉遵守模板"的效果 |
| **G2** | + Skills + Hooks | "强制 gate"的增益 |
| **G3** | + Skills + Hooks + MCP | "数据流工具"的增益 |

**消融逻辑：**
- G1 vs G0 → 协议本身有没有用
- G2 vs G1 → 强制执行 vs 自觉遵守
- G3 vs G2 → 工具辅助的独立贡献

### 6.3 评估指标

#### 主指标（结果层面）

| 指标 | 定义 | 来源 |
|------|------|------|
| **Resolve Rate** | 任务测试全部通过 | SWE-bench 官方 |
| **Anti-Cheat Pass Rate** | 通过反投机规则检查 | 自定义规则扫描 |

#### 过程指标（支撑"更深入"）

| 指标 | 定义 | 测量方式 |
|------|------|----------|
| **Premature Test Ratio** | 无 intent/prediction 的测试比例 | trace.jsonl 统计 |
| **Evidence Citation Rate** | state.json 引用 MCP 证据的比例 | 自动检查字段 |
| **Patch Locality** | 补丁位置 vs root cause 位置 | 代码距离计算 |
| **Hypothesis Specificity** | intent/prediction 的具体性 | 人工标注 (抽样) |
| **Branch Convergence** | 假设收敛速度 | history 长度分析 |

### 6.4 反投机规则

```python
# anti_cheat.py 检测的模式
CHEAT_PATTERNS = [
    r"except.*:\s*pass",           # 吞异常
    r"except.*:\s*return\s+None",  # 静默返回
    r"@pytest.mark.skip",          # 跳过测试
    r"assert\s+True",              # 无意义断言
]
```

### 6.5 作为 Empirical Study 的研究问题

无论结果如何，可以回答：

| 研究问题 | 价值 |
|----------|------|
| RQ1: Agent 在多步调试中的典型失败模式是什么？ | 行为分类 taxonomy |
| RQ2: "先测试再想"的现象有多普遍？ | Premature Test Ratio 量化 |
| RQ3: Prompt 约束 vs Runtime 约束的效果差异？ | G1 vs G2 对比 |
| RQ4: 数据流工具能否改善复杂任务表现？ | G3 vs G2 对比 |

---

## 第七部分：分阶段实施计划

### Phase 0：Pilot 验证

| 任务 | 目的 |
|------|------|
| 手选 10 个任务（5 简单 + 5 复杂） | 快速验证 |
| 只对比 G0 vs G2 | 看有没有明显差异 |
| 人工观察 agent 行为 | 定性理解"懒"的表现 |

### Phase 1：Skills + Hooks（最小可验证版本）

**目标：** 验证"强制结构化约束"本身是否有效

| 任务 | 产出 | 预估时间 |
|------|------|----------|
| 设计 Skill 模板 | `debugging.skill.md` | 1 天 |
| 实现 pytest_gate.py | Hook 脚本 | 1 天 |
| 实现 post_test.py | Hook 脚本 + trace.jsonl | 1 天 |
| 搭建 SWE-bench 运行环境 | 可复现的实验脚本 | 2 天 |
| 跑 G0 vs G2 对比 | 初步数据 | 3 天 |

**Phase 1 完成标准：**
- G0/G2 在 Lite 上各跑 50 个任务
- 能计算 Resolve Rate 和 Premature Test Ratio
- 观察到可解释的差异

### Phase 2：加入 MCP

**目标：** 验证"数据流工具"的独立贡献

| 任务 | 产出 | 预估时间 |
|------|------|----------|
| 实现 Jedi adapter | trace_symbol() | 2 天 |
| 实现 PyCG adapter | trace_callchain() | 2 天 |
| 封装 MCP Server | 可运行的 MCP | 1 天 |
| 更新 Skill 模板 | 引导 agent 使用 MCP | 1 天 |
| 跑 G3 实验 | 完整四组数据 | 3 天 |

**Phase 2 完成标准：**
- G0/G1/G2/G3 全量对比
- 能计算 Evidence Citation Rate
- G3 vs G2 有可解释的增益（或有意义的负结果）

### Phase 3：强验收 + 论文

**目标：** 可信结论 + 论文撰写

| 任务 | 产出 | 预估时间 |
|------|------|----------|
| Verified 子集实验 | 高可信度数据 | 3 天 |
| Patch Locality 分析 | 支撑"更深入"的证据 | 2 天 |
| Hypothesis Specificity 标注 | 抽样人工标注 | 2 天 |
| 论文撰写 | 初稿 | 5 天 |

---

## 第八部分：风险与局限性

### 8.1 技术风险

| 风险 | 概率 | 缓解措施 |
|------|------|----------|
| Hooks 拦截不稳定 | 低 | Pilot 阶段验证 |
| Jedi/PyCG 对复杂代码失效 | 中 | 记录失败案例，标注为 "tool limitation" |
| Agent 形式化填写 state.json | 高 | 接受局限，用间接指标评估 |
| SWE-bench 任务不够复杂 | 中 | 抽样分析任务复杂度分布 |

### 8.2 研究局限性（论文需明确写出）

1. **语言限制**：仅验证 Python，多语言泛化是 future work
2. **Agent 限制**：仅测试 Claude，其他 LLM 可能表现不同
3. **形式化遵从**：无法区分"真推理"vs"模板填充"，用间接指标部分缓解
4. **Benchmark 限制**：SWE-bench 是 bug fix，不代表所有 coding 任务

### 8.3 预期贡献总结

| 类型 | 贡献 |
|------|------|
| **Empirical** | 首次系统分析 LLM agent 在多步调试中的推理行为 |
| **Methodology** | Skills + Hooks + MCP 三层约束框架 |
| **Tooling** | 可复用的 pytest gate、trace 工具、MCP server |
| **Insights** | Prompt 约束 vs Runtime 约束的效果对比 |

---

## 附录：关键决策记录

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 研究问题定位 | "结构化约束效果" + 间接证据支撑"更深入" | 平衡可声称性与可测量性 |
| MCP 实现方式 | Jedi + PyCG（现有框架） | 可行性优先于完美 |
| 语言范围 | Python-only | SWE-bench 契合，论文 scope 清晰 |
| 数据集 | SWE-bench (Lite + Verified) | 聚焦 bug fix，假设更契合 |
| 负结果策略 | 可转为 empirical study | 无论结果都有价值 |
