import time
import torch

DURATION = 10  # seconds per benchmark
SIZE = 2048     # matrix dimension


def check_devices():
    devices = ["cpu"]
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        devices.append("xpu")
        print(f"XPU device: {torch.xpu.get_device_name(0)}")
    else:
        print("WARNING: XPU not available, only CPU will be benchmarked.")
    print(f"PyTorch: {torch.__version__}")
    print(f"Devices: {devices}\n")
    return devices


def sync(device: str):
    if device == "xpu":
        torch.xpu.synchronize()


def bench_matmul(device: str) -> int:
    a = torch.randn(SIZE, SIZE, device=device, dtype=torch.float32)
    b = torch.randn(SIZE, SIZE, device=device, dtype=torch.float32)
    sync(device)
    count = 0
    end = time.perf_counter() + DURATION
    while time.perf_counter() < end:
        torch.mm(a, b)
        sync(device)
        count += 1
    return count


def bench_elementwise(device: str) -> int:
    a = torch.randn(SIZE, SIZE, device=device, dtype=torch.float32)
    b = torch.randn(SIZE, SIZE, device=device, dtype=torch.float32)
    sync(device)
    count = 0
    end = time.perf_counter() + DURATION
    while time.perf_counter() < end:
        torch.add(a, b)
        torch.mul(a, b)
        torch.sub(a, b)
        sync(device)
        count += 1
    return count


def bench_reduction(device: str) -> int:
    a = torch.randn(SIZE, SIZE, device=device, dtype=torch.float32)
    sync(device)
    count = 0
    end = time.perf_counter() + DURATION
    while time.perf_counter() < end:
        torch.sum(a)
        torch.mean(a)
        torch.std(a)
        sync(device)
        count += 1
    return count


def bench_conv2d(device: str) -> int:
    conv = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False).to(device)
    x = torch.randn(16, 64, 64, 64, device=device, dtype=torch.float32)
    sync(device)
    count = 0
    end = time.perf_counter() + DURATION
    while time.perf_counter() < end:
        conv(x)
        sync(device)
        count += 1
    return count


def bench_fft(device: str) -> int:
    a = torch.randn(SIZE, SIZE, device=device, dtype=torch.float32)
    sync(device)
    count = 0
    end = time.perf_counter() + DURATION
    while time.perf_counter() < end:
        torch.fft.fft2(a)
        sync(device)
        count += 1
    return count


def bench_svd(device: str) -> int:
    a = torch.randn(1024, 1024, device=device, dtype=torch.float32)
    sync(device)
    count = 0
    end = time.perf_counter() + DURATION
    while time.perf_counter() < end:
        torch.linalg.svd(a, full_matrices=False)
        sync(device)
        count += 1
    return count


BENCHMARKS = [
    ("MatMul 2048x2048",        bench_matmul),
    ("Elementwise add/mul/sub", bench_elementwise),
    ("Reduction sum/mean/std",  bench_reduction),
    ("Conv2d 64->128 (16x64²)", bench_conv2d),
    ("FFT2 2048x2048",         bench_fft),
    ("SVD 1024x1024",          bench_svd),
]


def main():
    devices = check_devices()
    results = {}  # {bench_name: {device: count}}

    for name, fn in BENCHMARKS:
        results[name] = {}
        for dev in devices:
            print(f"  [{dev.upper():>3}] {name} ...", end="", flush=True)
            # warm-up: run a few iterations first
            try:
                warm_a = torch.randn(128, 128, device=dev)
                sync(dev)
                del warm_a
                iters = fn(dev)
            except Exception as e:
                print(f" SKIPPED ({e})")
                iters = None
                continue
            results[name][dev] = iters
            print(f" {iters} iterations")

    # --- summary table ---
    print("\n" + "=" * 70)
    print(f"{'Benchmark':<28} {'CPU':>10} {'XPU':>10} {'Speedup':>10}")
    print("-" * 70)
    for name, _ in BENCHMARKS:
        row = results.get(name, {})
        cpu_n = row.get("cpu")
        xpu_n = row.get("xpu")
        cpu_s = str(cpu_n) if cpu_n is not None else "N/A"
        xpu_s = str(xpu_n) if xpu_n is not None else "N/A"
        if cpu_n and xpu_n:
            speedup = f"{xpu_n / cpu_n:.2f}x"
        else:
            speedup = "—"
        print(f"{name:<28} {cpu_s:>10} {xpu_s:>10} {speedup:>10}")
    print("=" * 70)
    print(f"Each benchmark ran for {DURATION}s. Higher iteration count = faster.\n")


if __name__ == "__main__":
    main()
