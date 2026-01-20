# Think More SWE-bench 测试计划

> 日期: 2025-01-20
> 版本: v1.0

## 目标

验证 Think More 在 SWE-bench Lite 数据集上的运行效果，确保：
1. Docker 容器集成正常工作
2. Claude Code + Think More hooks 正确执行
3. trace.jsonl 和 patch.diff 正确输出到宿主机

---

## 测试范围

### Phase 1: 小规模验证 (10 实例)

| 实例 ID | 仓库 | 说明 |
|---------|------|------|
| astropy__astropy-12907 | astropy | 天文学库 |
| astropy__astropy-14182 | astropy | 天文学库 |
| astropy__astropy-14365 | astropy | 天文学库 |
| astropy__astropy-14995 | astropy | 天文学库 |
| astropy__astropy-6938 | astropy | 天文学库 |
| astropy__astropy-7746 | astropy | 天文学库 |
| django__django-10914 | django | Web 框架 |
| django__django-10924 | django | Web 框架 |
| django__django-11001 | django | Web 框架 |
| django__django-11019 | django | Web 框架 |

### Phase 2: 全量测试 (300 实例)

待 Phase 1 验证通过后执行。

---

## 执行步骤

### Step 1: 构建 Think More 镜像

```bash
cd /home/zhihao/hdd/think_more

# 构建前 10 个实例的镜像
python docker/build_agent_images_parallel.py 10

# 验证镜像
docker images | grep think-more | head -10
```

**预期结果:**
- 10 个 `xxx-think-more:latest` 镜像构建成功
- 每个镜像包含 Claude Code + Think More hooks

**预计时间:** ~10-15 分钟（取决于网络速度）

---

### Step 2: 单实例测试

```bash
# 设置环境变量
export ANTHROPIC_API_KEY="your-api-key"
export CLAUDE_MODEL="sonnet"

# 运行单个实例
python experiments/runner.py django__django-11099 run_full
```

**预期结果:**
- `output/swebenchlite/claude_code_think_more/run_full/django__django-11099/`
  - `trace.jsonl` - Claude Code 执行 trace
  - `patch.diff` - 生成的代码修改
  - `prompt.txt` - 输入的 prompt
  - `result.json` - 结果摘要

**验证点:**
1. 容器正常启动和退出
2. Think More hooks 被调用（查看 trace.jsonl）
3. patch.diff 不为空

**预计时间:** ~5-20 分钟/实例

---

### Step 3: 批量测试 (10 实例)

```bash
# 运行 10 实例测试脚本
./experiments/run_10_instances.sh run_full 2

# 或手动运行
python experiments/batch_runner.py experiments/instances_10.txt run_full 2 2 claude_code_think_more 1200
```

**参数说明:**
- `run_full`: 无限制执行模式
- `2`: k 值（run_less 模式用）
- `2`: 并发数
- `1200`: 超时时间（秒）

**预期结果:**
- 10 个实例的输出目录
- 每个包含 trace.jsonl 和 patch.diff

---

## 验证清单

### 镜像构建验证

- [ ] `docker images | grep think-more` 显示 10 个镜像
- [ ] 每个镜像大小合理（~2-3GB）
- [ ] 无构建错误（检查 `docker/build_logs/`）

### 单实例验证

- [ ] 容器正常启动（无权限错误）
- [ ] Claude Code 正确连接 API
- [ ] trace.jsonl 包含 Think More hooks 事件
- [ ] patch.diff 包含有效的 git diff
- [ ] 执行时间在合理范围（5-20 分钟）

### 批量验证

- [ ] 所有 10 个实例完成
- [ ] checkpoint.json 正确更新
- [ ] 无并发冲突
- [ ] 总成功率统计

---

## 输出目录结构

```
think_more/
└── output/
    └── swebenchlite/
        └── claude_code_think_more/
            └── run_full/
                ├── checkpoint.json
                ├── astropy__astropy-12907/
                │   ├── prompt.txt
                │   ├── trace.jsonl
                │   ├── patch.diff
                │   └── result.json
                ├── django__django-11099/
                │   └── ...
                └── ...
```

---

## 问题排查

### 常见问题

1. **API Key 未设置**
   ```
   Error: Missing ANTHROPIC_API_KEY
   ```
   解决: `export ANTHROPIC_API_KEY="your-key"`

2. **Docker 镜像未找到**
   ```
   Error: No Docker image found for instance
   ```
   解决: 运行 `python docker/build_agent_images_parallel.py`

3. **超时**
   ```
   Error: Timeout
   ```
   解决: 增加 timeout 参数，或检查网络

4. **Think More hooks 未执行**
   检查: `grep -l "hook" output/.../trace.jsonl`
   解决: 检查 `docker/configure_models.sh` 中的 hooks 配置

---

## 后续计划

### Phase 1 完成后

1. 分析 10 实例的结果
2. 检查 Think More 对执行次数的影响
3. 对比 run_free vs run_full 模式

### Phase 2

1. 扩展到全部 300 个 SWE-bench Lite 实例
2. 并行度提升到 4-8
3. 收集完整的指标数据

---

## 执行记录

| 时间 | 操作 | 结果 | 备注 |
|------|------|------|------|
| | | | |

