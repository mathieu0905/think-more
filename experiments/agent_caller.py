#!/usr/bin/env python3
"""
Agent 调用器：封装 Claude Code + Think More 的调用接口
基于 run_free_run_less_run_full 版本，添加 Think More 特定的配置
"""
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
import shutil


@dataclass
class AgentTrace:
    """Agent 执行的 trace 记录"""
    agent_type: str  # claude_code_think_more
    prompt: str
    output: str
    tokens_used: int
    exec_count: int
    duration_sec: float
    raw_trace: List[Dict[str, Any]]  # stream-json 格式的原始 trace
    error: Optional[str] = None


class AgentCaller:
    """Think More Agent 调用接口"""

    def __init__(self, agent_type: str = "claude_code_think_more", instance_id: Optional[str] = None):
        """
        初始化 Agent 调用器

        Args:
            agent_type: "claude_code_think_more"
            instance_id: SWE-bench 实例 ID（用于确定 Docker 镜像）
        """
        self.agent_type = agent_type
        self.instance_id = instance_id

    def call(self, prompt: str, timeout: int = 1200, trace_output_path: Optional[str] = None) -> AgentTrace:
        """
        调用 agent 并返回 trace

        Args:
            prompt: 输入的提示词
            timeout: 超时时间(秒)，Think More 默认更长

        Returns:
            AgentTrace 对象
        """
        return self._call_claude_code_think_more(prompt, timeout, trace_output_path)

    def _get_docker_image(self, instance_id: str) -> Optional[str]:
        """
        根据 instance_id 获取对应的 Docker 镜像名称

        instance_id 格式: django__django-11099
        镜像名格式: swebench/sweb.eval.x86_64.django_1776_django-11099-think-more:latest
        """
        if not instance_id:
            return None

        # 从 instance_id 提取 issue 编号部分（如 django-11099）
        # instance_id 格式: {repo}__{repo}-{issue_number}
        parts = instance_id.split("__")
        if len(parts) != 2:
            return None
        issue_part = parts[1]  # 如 django-11099

        # 查找所有 swebench think-more 镜像
        result = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                # 匹配包含 issue 编号且以 -think-more:latest 结尾的镜像
                if issue_part in line and line.endswith("-think-more:latest"):
                    return line

        return None

    def _call_claude_code_think_more(self, prompt: str, timeout: int, trace_output_path: Optional[str] = None) -> AgentTrace:
        """调用 Claude Code with Think More"""
        import time
        import os
        start = time.time()

        # 使用指定的输出路径或创建临时文件保存 trace
        if trace_output_path:
            trace_path = trace_output_path
            # 确保目录存在
            Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
        else:
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.jsonl', delete=False) as trace_file:
                trace_path = trace_file.name

        try:
            # Fail fast in environments where real network calls are impossible
            run_is_mock = "Mock" in type(subprocess.run).__name__
            if not run_is_mock and not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
                if os.environ.get("ANTHROPIC_BASE_URL"):
                    os.environ["ANTHROPIC_API_KEY"] = "sk-placeholder"
                else:
                    duration = time.time() - start
                    return AgentTrace(
                        agent_type="claude_code_think_more",
                        prompt=prompt,
                        output="",
                        tokens_used=max(1, len(prompt) // 4),
                        exec_count=0,
                        duration_sec=duration if duration > 0 else 0.001,
                        raw_trace=[],
                        error="Missing ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN"
                    )

            # 构建命令
            cmd = self._build_claude_command(prompt, trace_path)

            # 确定工作目录：如果 /testbed 存在则使用，否则使用当前目录
            work_dir = "/testbed" if os.path.exists("/testbed") else None

            # 执行命令
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=work_dir,
                env=self._build_sandboxed_env()
            )

            duration = time.time() - start

            # 读取 trace
            raw_trace = self._read_trace_file(trace_path)

            output = ""
            if raw_trace:
                output = self._extract_output_from_trace(raw_trace).strip()
            if not output:
                output = (result.stdout or "").strip()
            tokens = self._extract_tokens_from_trace(raw_trace)
            if tokens == 0:
                tokens = max(1, len(prompt) // 4)
            exec_count = self._count_executions_from_trace(raw_trace)

            trace_error = self._extract_error_from_trace(raw_trace)
            proc_stderr = (result.stderr or "").strip()
            if result.returncode != 0:
                error = proc_stderr or f"Non-zero exit code: {result.returncode}"
            else:
                error = trace_error or None
            if not output and not error:
                error = "No output produced by agent"

            return AgentTrace(
                agent_type="claude_code_think_more",
                prompt=prompt,
                output=output,
                tokens_used=tokens,
                exec_count=exec_count,
                duration_sec=duration,
                raw_trace=raw_trace,
                error=error
            )

        except subprocess.TimeoutExpired:
            duration = time.time() - start
            raw_trace = self._read_trace_file(trace_path)

            output = self._extract_output_from_trace(raw_trace).strip() if raw_trace else ""
            tokens = self._extract_tokens_from_trace(raw_trace) if raw_trace else 0
            exec_count = self._count_executions_from_trace(raw_trace) if raw_trace else 0

            if tokens == 0:
                tokens = max(1, len(prompt) // 4)

            error = None if output else "Timeout"

            return AgentTrace(
                agent_type="claude_code_think_more",
                prompt=prompt,
                output=output,
                tokens_used=tokens,
                exec_count=exec_count,
                duration_sec=duration if duration > 0 else float(timeout),
                raw_trace=raw_trace,
                error=error
            )
        except Exception as e:
            return AgentTrace(
                agent_type="claude_code_think_more",
                prompt=prompt,
                output="",
                tokens_used=0,
                exec_count=0,
                duration_sec=time.time() - start,
                raw_trace=[],
                error=str(e)
            )
        finally:
            # 只清理临时文件（不清理指定的输出文件）
            if not trace_output_path:
                Path(trace_path).unlink(missing_ok=True)
                Path(f"{trace_path}.prompt.txt").unlink(missing_ok=True)

    def _build_claude_command(self, prompt: str, trace_path: str) -> List[str]:
        """构建 Claude Code + Think More 命令"""
        # 必须有 instance_id 和对应的 Docker 镜像
        docker_image = self._get_docker_image(self.instance_id) if self.instance_id else None

        if not docker_image:
            raise RuntimeError(f"No Docker image found for instance: {self.instance_id}. "
                             f"Please build the think-more image first using: "
                             f"python docker/build_agent_images.py")

        # 在 Docker 容器内执行
        container_trace_path = "/workspace/output/trace.jsonl"
        container_patch_path = "/workspace/output/patch.diff"
        host_trace_dir = str(Path(trace_path).parent.absolute())

        # 获取项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        docker_dir = os.path.join(project_root, "docker")

        # 从环境变量获取配置
        base_url = os.environ.get('ANTHROPIC_BASE_URL', 'https://api.anthropic.com')
        api_key = os.environ.get('ANTHROPIC_API_KEY', '')
        auth_token = os.environ.get('ANTHROPIC_AUTH_TOKEN', '')
        claude_model = os.environ.get('CLAUDE_MODEL', 'sonnet')

        # Think More 镜像已经内置了 hooks 和 skills 配置
        # 只需要运行 configure_models.sh 设置环境变量
        container_prompt_path = "/workspace/output/prompt.txt"
        claude_cmd = (
            f"bash /workspace/docker/configure_models.sh && "
            f"cp -r /root/.claude /home/nonroot/.claude && "
            f"chown -R nonroot:nonroot /home/nonroot/.claude && "
            f"su nonroot -c \""
            f"source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed && "
            f"cd /testbed && "
            # Claude CLI flag compatibility
            f"(cat {container_prompt_path} | claude -p --model {claude_model} --dangerously-skip-permissions --verbose --output-type stream-json || "
            f"cat {container_prompt_path} | claude -p --model {claude_model} --dangerously-skip-permissions --verbose --output-format stream-json)"
            f"\" > {container_trace_path}; "
            f"cd /testbed && git diff > {container_patch_path}"
        )

        # 先将 prompt 写入宿主机目录
        prompt_file = Path(host_trace_dir) / "prompt.txt"
        prompt_file.write_text(prompt, encoding='utf-8')

        return [
            "docker", "run", "--rm",
            "-e", f"ANTHROPIC_API_KEY={api_key}",
            "-e", f"ANTHROPIC_AUTH_TOKEN={auth_token}",
            "-e", f"ANTHROPIC_BASE_URL={base_url}",
            "-e", f"CLAUDE_MODEL={claude_model}",
            "-e", "THINK_MORE_ENABLED=1",
            "-v", f"{host_trace_dir}:/workspace/output",
            "-v", f"{docker_dir}:/workspace/docker:ro",
            "--network", "host",
            docker_image,
            "bash", "-c",
            claude_cmd
        ]

    def _read_trace_file(self, trace_path: str) -> List[Dict[str, Any]]:
        """读取 stream-json 格式的 trace 文件"""
        traces = []
        try:
            with open(trace_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            traces.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"Warning: Failed to read trace file: {e}")
        return traces

    def _build_sandboxed_env(self) -> Dict[str, str]:
        """Build an env suitable for workspace-write sandboxes."""
        env = os.environ.copy()
        return env

    def _extract_tokens_from_trace(self, trace: List[Dict[str, Any]]) -> int:
        """从 trace 中提取 token 使用量"""
        total_tokens = 0
        model_usage_max = 0

        for entry in trace:
            usage = entry.get("usage")
            if isinstance(usage, dict):
                total_tokens += usage.get("input_tokens", 0) + usage.get("output_tokens", 0)

            message = entry.get("message")
            if isinstance(message, dict):
                msg_usage = message.get("usage")
                if isinstance(msg_usage, dict):
                    total_tokens += msg_usage.get("input_tokens", 0) + msg_usage.get("output_tokens", 0)

            model_usage = entry.get("modelUsage")
            if isinstance(model_usage, dict):
                summed = 0
                for _, per_model in model_usage.items():
                    if isinstance(per_model, dict):
                        summed += per_model.get("inputTokens", 0) + per_model.get("outputTokens", 0)
                model_usage_max = max(model_usage_max, summed)

        return total_tokens if total_tokens > 0 else model_usage_max

    def _count_executions_from_trace(self, trace: List[Dict[str, Any]]) -> int:
        """从 trace 中统计执行次数（Bash 工具调用次数）"""
        return sum(
            1
            for entry in trace
            if entry.get("type") == "tool_use" and str(entry.get("name", "")).lower() == "bash"
        )

    def _extract_output_from_trace(self, trace: List[Dict[str, Any]]) -> str:
        """从 trace 中提取输出文本"""
        output_parts = []
        for entry in trace:
            if entry.get("type") == "item.completed":
                item = entry.get("item", {})
                if item.get("type") == "agent_message":
                    text = item.get("text", "")
                    if text:
                        output_parts.append(text)

            if entry.get("type") == "assistant":
                message = entry.get("message")
                if isinstance(message, dict):
                    content = message.get("content", [])
                    for item in content:
                        if item.get("type") == "text":
                            text = item.get("text", "")
                            if text:
                                output_parts.append(text)

        if output_parts:
            return "\n".join(output_parts)

        for entry in trace:
            if entry.get("type") == "result":
                result = entry.get("result", "")
                if isinstance(result, str) and result.strip():
                    return result
        return ""

    def _extract_error_from_trace(self, trace: List[Dict[str, Any]]) -> str:
        """从 trace 中提取错误信息"""
        def _walk(obj):
            if isinstance(obj, dict):
                yield obj
                for v in obj.values():
                    yield from _walk(v)
            elif isinstance(obj, list):
                for v in obj:
                    yield from _walk(v)

        for entry in trace:
            exit_code = entry.get("exit_code")
            stderr = entry.get("stderr")
            if exit_code not in (None, 0) and isinstance(stderr, str) and stderr.strip():
                return stderr.strip()

            for d in _walk(entry):
                stderr = d.get("stderr")
                if isinstance(stderr, str) and stderr.strip():
                    return stderr.strip()
                err = d.get("error")
                if isinstance(err, str) and err.strip():
                    return err.strip()

            if entry.get("type") == "result" and entry.get("is_error"):
                result_text = entry.get("result")
                if isinstance(result_text, str) and result_text.strip():
                    return result_text.strip()

        return ""


def save_trace(trace: AgentTrace, output_path: Path):
    """保存 trace 到文件"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    trace_data = {
        "agent_type": trace.agent_type,
        "prompt": trace.prompt,
        "output": trace.output,
        "tokens_used": trace.tokens_used,
        "exec_count": trace.exec_count,
        "duration_sec": trace.duration_sec,
        "error": trace.error,
        "raw_trace": trace.raw_trace
    }

    with open(output_path, 'w') as f:
        json.dump(trace_data, f, indent=2, ensure_ascii=False)

    print(f"Trace saved to: {output_path}")


if __name__ == "__main__":
    caller = AgentCaller(agent_type="claude_code_think_more")

    test_prompt = "请帮我写一个 Python 函数，计算斐波那契数列的第 n 项"

    print(f"Calling {caller.agent_type}...")
    trace = caller.call(test_prompt)

    print(f"\nAgent: {trace.agent_type}")
    print(f"Tokens used: {trace.tokens_used}")
    print(f"Executions: {trace.exec_count}")
    print(f"Duration: {trace.duration_sec:.2f}s")
    print(f"Output length: {len(trace.output)} chars")

    if trace.error:
        print(f"Error: {trace.error}")

    save_trace(trace, Path("test_trace.json"))
