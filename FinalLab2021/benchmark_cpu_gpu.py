#!/usr/bin/env python3
"""Run the CPU and CUDA renderers with identical settings and print a report."""
import argparse
import re
import subprocess
import sys
from pathlib import Path


def run(command, log_path):
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=command[0].rsplit("/", 1)[0],
                                stdout=log, stderr=subprocess.STDOUT,
                                text=True, check=False)
    text = log_path.read_text(encoding="utf-8")
    match = re.search(r"BENCHMARK (\w+) .*?render_ms=([0-9.]+).*?total_ms=([0-9.]+)", text)
    if result.returncode != 0 or not match:
        print(f"Benchmark failed; see {log_path}", file=sys.stderr)
        sys.exit(result.returncode or 1)
    values = {"renderer": match.group(1), "render_ms": float(match.group(2)),
              "total_ms": float(match.group(3))}
    kernel = re.search(r"gpu_kernel_ms=([0-9.]+)", text)
    if kernel:
        values["gpu_kernel_ms"] = float(kernel.group(1))
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", default="build-cuda")
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--material", choices=("diffuse", "microfacet"), default="diffuse")
    parser.add_argument("--mode", choices=("mis", "sppm"), default="mis")
    parser.add_argument("--log-dir", default="benchmark-results")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    build = (root / args.build).resolve()
    logs = (root / args.log_dir).resolve()
    logs.mkdir(parents=True, exist_ok=True)
    cpu_output, gpu_output = logs / "cpu.ppm", logs / "gpu.ppm"
    common = [str(args.iterations), str(args.width), str(args.height), str(args.max_depth)]
    cpu = run([str(build / "RayTracing"), *common, args.material, str(cpu_output)], logs / "cpu.log")
    gpu = run([str(build / "CudaPathTracing"), *common, args.material, args.mode, str(gpu_output)], logs / "gpu.log")
    speedup = cpu["total_ms"] / gpu["total_ms"] if gpu["total_ms"] else float("inf")
    print("CPU vs GPU benchmark")
    print(f"settings: {args.width}x{args.height}, iterations={args.iterations}, depth={args.max_depth}, material={args.material}, mode={args.mode}")
    print(f"CPU: total={cpu['total_ms']:.3f} ms, render={cpu['render_ms']:.3f} ms")
    extra = f", gpu_kernel={gpu['gpu_kernel_ms']:.3f} ms" if "gpu_kernel_ms" in gpu else ""
    print(f"GPU: total={gpu['total_ms']:.3f} ms, render={gpu['render_ms']:.3f} ms{extra}")
    print(f"speedup (CPU total / GPU total): {speedup:.2f}x")
    print(f"logs and images: {logs}")


if __name__ == "__main__":
    main()