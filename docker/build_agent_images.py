#!/usr/bin/env python3
"""
构建 Think More per-repo overlay 镜像

从 SWE-bench 数据集读取实例，查找对应的基础镜像，
构建带有 Think More 配置的 overlay 镜像。
"""
import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple
from datasets import load_dataset


def get_available_base_images() -> dict:
    """获取所有可用的 SWE-bench 基础镜像"""
    result = subprocess.run(
        ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
        capture_output=True,
        text=True
    )

    images = {}
    if result.returncode == 0:
        for line in result.stdout.strip().split('\n'):
            if line.startswith("swebench/sweb.eval.x86_64."):
                # 提取 issue 部分作为 key
                # 格式: swebench/sweb.eval.x86_64.django_1776_django-11099:latest
                images[line] = line

    return images


def find_base_image(instance_id: str, available_images: dict) -> Optional[str]:
    """
    根据 instance_id 查找对应的基础镜像

    instance_id 格式: django__django-11099
    基础镜像格式: swebench/sweb.eval.x86_64.django_1776_django-11099:latest
    """
    parts = instance_id.split("__")
    if len(parts) != 2:
        return None

    issue_part = parts[1]  # 如 django-11099

    for image_name in available_images:
        if issue_part in image_name and ":latest" in image_name:
            # 排除已经是 think-more 的镜像
            if "-think-more:" not in image_name:
                return image_name

    return None


def build_think_more_image(base_image: str, instance_id: str, project_root: Path, force: bool = False) -> Tuple[bool, str]:
    """
    构建 Think More overlay 镜像

    Args:
        base_image: 基础镜像名称
        instance_id: SWE-bench 实例 ID
        project_root: 项目根目录
        force: 是否强制重新构建

    Returns:
        (success, message) 元组
    """
    # 提取 issue 部分作为镜像名
    parts = instance_id.split("__")
    issue_part = parts[1]  # 如 django-11099

    # 目标镜像名
    target_image = f"{base_image.replace(':latest', '')}-think-more:latest"

    # 检查是否已存在
    if not force:
        result = subprocess.run(
            ["docker", "images", "-q", target_image],
            capture_output=True,
            text=True
        )
        if result.stdout.strip():
            return True, f"已存在: {target_image}"

    # 构建镜像
    dockerfile = project_root / "docker" / "Dockerfile.think-more"

    cmd = [
        "docker", "build",
        "--build-arg", f"BASE_IMAGE={base_image}",
        "-t", target_image,
        "-f", str(dockerfile),
        str(project_root)
    ]

    print(f"构建: {target_image}")
    print(f"  基础镜像: {base_image}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        return True, f"构建成功: {target_image}"
    else:
        return False, f"构建失败: {result.stderr[:500]}"


def get_first_n_instances(n: int, dataset_name: str = "princeton-nlp/SWE-bench_Lite") -> List[str]:
    """获取数据集的前 N 个实例 ID"""
    dataset = load_dataset(dataset_name, split="test")
    instance_ids = [item["instance_id"] for item in dataset]
    return instance_ids[:n]


def main():
    parser = argparse.ArgumentParser(description="构建 Think More per-repo overlay 镜像")
    parser.add_argument("--count", type=int, default=10, help="要构建的实例数量 (默认: 10)")
    parser.add_argument("--instance", type=str, help="指定单个实例 ID")
    parser.add_argument("--instances-file", type=str, help="从文件读取实例 ID 列表")
    parser.add_argument("--force", action="store_true", help="强制重新构建已存在的镜像")
    parser.add_argument("--dataset", type=str, default="princeton-nlp/SWE-bench_Lite", help="数据集名称")
    parser.add_argument("--list-only", action="store_true", help="只列出需要构建的镜像，不实际构建")
    args = parser.parse_args()

    # 获取项目根目录
    project_root = Path(__file__).parent.parent

    # 获取可用的基础镜像
    print("正在扫描可用的 SWE-bench 基础镜像...")
    available_images = get_available_base_images()
    print(f"找到 {len(available_images)} 个基础镜像")

    # 确定要构建的实例列表
    if args.instance:
        instance_ids = [args.instance]
    elif args.instances_file:
        with open(args.instances_file) as f:
            instance_ids = [
                line.strip() for line in f
                if line.strip() and not line.strip().startswith("#")
            ]
    else:
        print(f"正在加载数据集前 {args.count} 个实例...")
        instance_ids = get_first_n_instances(args.count, args.dataset)

    print(f"将处理 {len(instance_ids)} 个实例")
    print("=" * 60)

    # 查找基础镜像并构建
    success_count = 0
    skip_count = 0
    fail_count = 0
    missing_base = []

    for i, instance_id in enumerate(instance_ids, 1):
        base_image = find_base_image(instance_id, available_images)

        if not base_image:
            print(f"[{i}/{len(instance_ids)}] ⚠ {instance_id}: 未找到基础镜像")
            missing_base.append(instance_id)
            continue

        if args.list_only:
            target = f"{base_image.replace(':latest', '')}-think-more:latest"
            print(f"[{i}/{len(instance_ids)}] {instance_id}")
            print(f"    基础: {base_image}")
            print(f"    目标: {target}")
            continue

        success, message = build_think_more_image(
            base_image, instance_id, project_root, args.force
        )

        if success:
            if "已存在" in message:
                print(f"[{i}/{len(instance_ids)}] ⏭ {instance_id}: {message}")
                skip_count += 1
            else:
                print(f"[{i}/{len(instance_ids)}] ✓ {instance_id}: {message}")
                success_count += 1
        else:
            print(f"[{i}/{len(instance_ids)}] ✗ {instance_id}: {message}")
            fail_count += 1

    # 打印总结
    print("=" * 60)
    print("构建总结:")
    print(f"  成功构建: {success_count}")
    print(f"  已存在跳过: {skip_count}")
    print(f"  构建失败: {fail_count}")
    print(f"  缺少基础镜像: {len(missing_base)}")

    if missing_base:
        print("\n缺少基础镜像的实例:")
        for inst in missing_base:
            print(f"  - {inst}")
        print("\n请先构建 SWE-bench 基础镜像:")
        print("  cd /path/to/SWE-bench")
        print("  python -m swebench.harness.run_evaluation --instances_path <dataset> --predictions_path <file> --run_id test")

    # 生成实例列表文件
    if not args.list_only and not args.instance:
        instances_file = project_root / "experiments" / f"instances_{len(instance_ids)}.txt"
        with open(instances_file, "w") as f:
            for inst in instance_ids:
                f.write(inst + "\n")
        print(f"\n实例列表已保存到: {instances_file}")

    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()
