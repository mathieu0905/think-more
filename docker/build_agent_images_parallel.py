#!/usr/bin/env python3
"""
并行构建 Think More overlay 镜像

从 Docker Hub pull 官方 SWE-bench 镜像，然后构建 Think More overlay。
"""
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DOCKERFILE = Path(__file__).parent / "Dockerfile.think-more"
IMAGE_LIST_LITE = Path(__file__).parent / "image_list.txt"
IMAGE_LIST_VERIFIED = Path(__file__).parent / "image_list_verified.txt"
LOG_DIR = Path(__file__).parent / "build_logs"
PROJECT_ROOT = Path(__file__).parent.parent
MAX_WORKERS = 8


def build_one(base_image: str) -> tuple[str, bool, str]:
    """Pull base image and build Think More overlay. Returns (image, success, msg)."""
    base_tag = f"{base_image}:latest"
    think_more_tag = f"{base_image}-think-more:latest"

    # Create log directory and file
    LOG_DIR.mkdir(exist_ok=True)
    log_file = LOG_DIR / f"{base_image.replace('/', '_').replace(':', '_')}.log"

    with open(log_file, "w", buffering=1) as log:
        log.write(f"Building {base_image}\n")
        log.write(f"Base tag: {base_tag}\n")
        log.write(f"Think More tag: {think_more_tag}\n\n")
        log.flush()

        # Check if think-more image already exists
        r = subprocess.run(["docker", "image", "inspect", think_more_tag], capture_output=True, text=True)
        if r.returncode == 0:
            log.write("Image already exists, skipping\n")
            log.flush()
            return base_image, True, "skipped"

        # Pull base image
        log.write("=" * 60 + "\n")
        log.write("PULLING BASE IMAGE\n")
        log.write("=" * 60 + "\n")
        log.flush()
        r = subprocess.run(["docker", "pull", base_tag], capture_output=True, text=True)
        log.write(r.stdout)
        log.write(r.stderr)
        log.flush()
        if r.returncode != 0:
            log.write(f"\nPull failed with exit code {r.returncode}\n")
            log.flush()
            return base_image, False, f"pull failed (see {log_file.name})"

        # Build Think More overlay
        log.write("\n" + "=" * 60 + "\n")
        log.write("BUILDING THINK MORE IMAGE\n")
        log.write("=" * 60 + "\n")
        log.flush()
        r = subprocess.run([
            "docker", "build",
            "--network", "host",
            "--build-arg", f"BASE_IMAGE={base_tag}",
            "-t", think_more_tag,
            "-f", str(DOCKERFILE),
            str(PROJECT_ROOT)
        ], capture_output=True, text=True)
        log.write(r.stdout)
        log.write(r.stderr)
        log.flush()
        if r.returncode != 0:
            log.write(f"\nBuild failed with exit code {r.returncode}\n")
            log.flush()
            return base_image, False, f"build failed (see {log_file.name})"

        log.write("\n" + "=" * 60 + "\n")
        log.write("BUILD SUCCESSFUL\n")
        log.write("=" * 60 + "\n")
        log.flush()

    return base_image, True, "ok"


def main():
    dataset = "lite"
    count = None

    # Parse arguments
    for arg in sys.argv[1:]:
        if arg.lower() in ["lite", "verified"]:
            dataset = arg.lower()
        elif arg.isdigit():
            count = int(arg)

    if dataset == "verified":
        image_list = IMAGE_LIST_VERIFIED
    else:
        image_list = IMAGE_LIST_LITE

    if not image_list.exists():
        print(f"Error: Image list not found: {image_list}")
        sys.exit(1)

    images = [l.strip() for l in image_list.read_text().splitlines() if l.strip()]

    # Limit count if specified
    if count:
        images = images[:count]

    total = len(images)
    print(f"Building {total} Think More images from {dataset} dataset with {MAX_WORKERS} workers...")
    print(f"Dockerfile: {DOCKERFILE}")
    print()

    success, skipped, failed = 0, 0, []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(build_one, img): img for img in images}
        for i, f in enumerate(as_completed(futures), 1):
            img, ok, msg = f.result()
            short = img.split(".")[-1]
            if ok:
                if msg == "skipped":
                    skipped += 1
                    print(f"[{i}/{total}] ⏭ {short} (already exists)")
                else:
                    success += 1
                    print(f"[{i}/{total}] ✓ {short}")
            else:
                failed.append((img, msg))
                print(f"[{i}/{total}] ✗ {short}: {msg}")

    print()
    print(f"Done: {success} built, {skipped} skipped, {len(failed)} failed")
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for img, msg in failed[:10]:
            print(f"  {img}: {msg}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")


if __name__ == "__main__":
    main()
