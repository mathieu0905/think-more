好，那我给你一份**“三件套融合版”的完整方案**：Skills（协议）+ Hooks（强制与观测）+ MCP（证据工具）**同时设计、彼此咬合**，而不是三个点各讲各的。

我会按你要求把 **可行性 / 数据集 / 测试验收** 全部一起纳入，形成一套能长期迭代、能严谨对比、能跑出可信结论的系统。

---

# 研究方案（融合 Skills + Hooks + MCP 的一体化设计）

## 总目标（你要解决的“懒”）

你说的“懒”不是不跑测试，而是：

> **明明可以先想清楚数据流与分支，形成可验证假设，再用测试验证；但 agent 往往跳过这一步，直接跑测试并进入边跑边猜。**

所以我们要实现的是：
**把“测试前思考质量”制度化**，并让数据流证据可得。

---

# A. 三层架构：协议层 / 控制层 / 证据层（必须同时存在）

## A1) Skills：协议层（让 agent “知道要怎么想”）

**Skill 的定位：定义“合格的测试前思考”长什么样。**
它不强制执行，但它提供模板和规范，保证行为可复用、可对齐。

**Skill 输出一个标准化“推理状态文件”（唯一真相源）**，比如：

* `DRP.md`（人类可读）
* 或 `state.json`（机器可读，后续统计方便）

> 我建议两者都要：`state.json` 给机器统计，`DRP.md` 给人看 trace。

### Skill 中必须规定的 3 件事（非常关键）

1. **Test Intent（测试意图）**：这次测试用来确认/排除哪个假设
2. **Prediction（预测）**：若结果 A/B 各意味着什么
3. **Update（证据回填）**：测试后必须更新假设并收敛分支

再加一个你最在意的：

4. **Dataflow Summary（数据流闭环最小版本）**：关键变量的 def→transform→use 链路（允许粗粒度，但必须写出来）

> Skill 只负责“写清楚格式”，不要负责“执法”，执法交给 Hooks。

---

## A2) Hooks：控制层（让 agent “必须按规则做” + 全程可观测）

Hooks 的定位是 **Policy Enforcement + Logging**，也就是：

### 1）测试前置条件（Test Gate）

你不是要“少跑测试”，你要的是：

> **每次跑测试都必须先写 intent + prediction**
> **每次跑完测试都必须写 update**

所以 Hook 该做的是：

* **PreToolUse（pytest 即将执行）**

  * 检查 `state.json` 是否有 `intent + prediction`
  * 没有就直接拦住（并把原因反馈给 agent）

* **PostToolUse（pytest 执行完成）**

  * 把测试结果摘要写入日志（pass/fail/stacktrace hash/耗时）
  * 并强制要求 agent 更新 `state.json` 的 `update` 字段（否则下一次 pytest 拦截）

这会把你的“懒”严格变成一个**无法发生的状态**：

> 不写推理结构 → 不能测试。
> 不做证据回填 → 不能进入下一轮测试。

### 2）观测与统计（Process Logging）

Hook 每一步都记录为 `trace.jsonl`：

* tool_name（bash/edit/…）
* command（pytest/rg/…）
* files_changed（git diff summary）
* whether_test_gate_passed
* state_version（state.json 的 hash）

这会让你后面所有过程指标（比如“过早测试比例”“无回填循环长度”“漂移次数”）都能**自动计算**，不靠主观。

> ⚠️ 现实可行性：如果你担心某些版本 hooks 触发不稳定，那就“双轨兜底”
>
> * hooks 做 gate
> * 同时 bash wrapper tee 输出做 backup log
>   这样不影响严谨性。

---

## A3) MCP：证据层（让 agent 在复杂分支下“想得动”）

MCP 的定位是 **给 agent 一个“数据流证据按钮”**。

你不需要一开始就搞重型静态分析框架；为了可行性，我建议 MCP **第一版只做 2 个工具**，但必须是“真的能让推理收敛”的那种：

### MCP 工具 1：`call_chain(symbol | failing_location)`

输出：

* 从 failing stacktrace 往上追的调用链摘要（函数→文件→行号）
* 只要 top-k（比如 30 行以内），让 agent 读得动

### MCP 工具 2：`def_use_slice(var, location)`

输出：

* 关键变量在相关文件内的 def/use 切片（简单版就行）
* 给出候选 def 点列表 + 每个 def 到 use 的路径提示

> 这两个工具足够支撑你想要的“先想清楚数据流再测”。

MCP 的产出会被要求写回 `state.json`：

* `evidence.mcp_callgraph`
* `evidence.mcp_defuse`
* `dataflow_chain`（从 MCP 证据抽象出的链路）

这样你后面还能量化：**模型是否真的使用了数据流证据**，不是摆设。

---

# B. 三者如何“咬合”：一个闭环工作流（真正融合点）

下面是你要的“融合后运行机制”，不是三块拼盘：

## Step 0：初始化状态（Skill触发）

Skill 让 agent先生成：

* hypotheses（最多 2 个）
* dataflow_chain（先写一个粗猜）
* next_probe（要验证什么）

## Step 1：证据获取（MCP优先）

当任务属于多步/多分支时（见后面的任务分层），策略规定：

* 优先调用 MCP（call_chain / def_use_slice）
* 把证据写入 state

## Step 2：测试前置（Hook执法）

当 agent 想跑 pytest：

* Hook 检查 state 是否具备 `intent + prediction`
* 不具备就拦住 → agent 必须补齐推理结构

## Step 3：测试执行（积极测试）

允许跑测试，且鼓励跑（你强调这是积极行为）：

* 但必须“带目的地跑”

## Step 4：证据回填（Hook执法）

pytest 结束后：

* Hook 记录结果
* 要求 agent 立刻更新 `update`：支持/排除哪条假设，下一步策略是什么

## Step 5：分支收敛（Skill约束）

Skill 规定：

* 候选假设最多 2 条
* 每轮测试后必须淘汰/确认至少 1 条
* 防止越跑越发散

---

# C. 可行性优先的“分阶段实现”（但仍然严谨）

你说“不需要最快”，但我们仍然要保证每一阶段都严谨、可运行、可对比。

## Phase 1：Skill + Hook（不做 MCP 也能跑）

目标：先把 **测试前思考制度化** 做稳

* Skill：state/DRP 模板
* Hook：pytest gate + update gate + trace logging

这一阶段就能验证你最核心的现象：
“baseline 在多步时会跳过推理结构，而 gate 后行为改变”。

## Phase 2：加 MCP（提升多分支收敛能力）

目标：让 dataflow 推理有“证据抓手”

* 实现 2 个 MCP 工具即可（call_chain + def_use_slice）
* state 里强制引用 MCP 证据（否则视为未使用）

---

# D. 数据集与任务选择：必须能触发多步与分支

为了严谨又可行，我建议 **Lite → Verified 子集** 两段式：

## D1) 开发与调参：SWE-bench Lite

* 成本低，适合机制迭代
* 足够多样，能触发多步行为

## D2) 强验收：SWE-bench Verified（抽样子集）

* 验收更可信（避免“过测试但投机”）
* 你不需要全跑，抽 20~50 个就能做强对照

---

# E. 测试验收：不仅“过了”，还要防投机（可行且不贵）

你会遇到：patch 过测试但本质是绕过逻辑（try/except 吞错等）。
所以我们做“两层验收”：

## E1) 第一层：官方测试验收（主指标）

* 任务测试全部通过
* 不新增失败

## E2) 第二层：反投机规则（轻量）

对 patch 做规则扫描（可自动）：

* 是否新增大范围 try/except 且吞异常
* 是否直接 skip/return 默认值绕过逻辑
* 是否大规模删测试或改断言
* 是否引入明显“禁用功能”的 hack

> 第二层不追求完美 correctness，但能显著减少“测试投机通过”的噪声。

Hooks 可以在 Stop 阶段做这层检查，并写入 trace，保证流程统一。

---

# F. 实验对照组（融合方案必须能消融归因）

你现在不急着写论文，但实验设计必须严谨可解释，所以组别要能回答：
到底是“多写解释（Skill）”有用，还是“强制 gate（Hook）”有用，还是“数据流工具（MCP）”有用。

我给你一套最干净、可跑的 4 组：

### G0 Baseline

原生 CC/Codex，不加任何约束

### G1 Skills only

只加 Skill（协议/模板），不拦截行为
→ 看“自觉”能改善多少

### G2 Skills + Hooks

加 Skill + Gate + Logging
→ 看“强制测试前思考”是否改变多步行为

### G3 Skills + Hooks + MCP

全量融合
→ 看“数据流证据工具”是否进一步提升复杂任务收敛

这就是你要的“三点融合”，同时具备可行性与可解释性。

---

# G. 你真正关心的“懒”怎么量化（与测试正相关而非负相关）

最后把指标写成符合你观点的版本：测试是积极行为，关键是**测试前后有没有推理闭环**。

### 核心指标（全部可由 hooks trace 自动统计）

1. **Premature Test Ratio**：无 intent/prediction 的测试比例
2. **Assimilation Rate**：测试后是否产生 hypothesis 更新与分支淘汰
3. **Branch Convergence Speed**：分支是否收敛、收敛步数
4. **Drift Rate**：假设前后矛盾、无证据跳变次数
5. **Patch Locality Proxy**：补丁更接近 def/transform 还是 use-site symptom fix

这些都不把“测试次数”当作坏事，而是把“测试是否成为证据”作为核心。

---

# 你现在得到的，是一套真正融合的“系统方案”

* **Skills** 定义“什么叫想清楚”（协议 + 模板）
* **Hooks** 保证“必须想清楚才能测、测完必须回填”（强制 + 全量日志）
* **MCP** 提供“想清楚所需的数据流证据”（call chain + def-use slice）

并且同时满足：

* ✅ 可行性：第一阶段不需要重型静态分析即可跑通；第二阶段仅 2 个 MCP 工具
* ✅ 数据集：Lite 定型，Verified 子集强验收
* ✅ 测试验收：官方测试 + 反投机规则双层验收

---

如果你下一步想继续“慢慢严谨”推进，我建议我们就从 **统一 state.json 的 schema** 开始（这是三者融合的锚点：Skill 写它、Hook 检它、MCP 填它）。
我可以直接给你一个**最小但够用**的 schema，让你后面实现不会走样。
