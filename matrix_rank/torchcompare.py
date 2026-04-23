import time
import torch
import torch.nn.functional as F


def _format_bytes(num_bytes: int) -> str:
    gb = num_bytes / (1024 ** 3)
    return f"{gb:.2f} GB"

DURATION = 10  # seconds per benchmark
BATCH_SIZE = 128
MATMUL_SIZE = 1024
ELEMENTWISE_SIZE = 512
REDUCTION_SIZE = 512
CONV_CHANNELS = 32
CONV_SPATIAL = 32
FFT_SIZE = 256
SVD_SIZE = 64


def op_matmul(a: torch.Tensor, b: torch.Tensor):
    return torch.bmm(a, b)


def op_elementwise(a: torch.Tensor, b: torch.Tensor):
    return torch.add(a, b), torch.mul(a, b), torch.sub(a, b)


def op_reduction(a: torch.Tensor):
    return torch.sum(a, dim=(1, 2)), torch.mean(a, dim=(1, 2)), torch.std(a, dim=(1, 2))


def op_conv2d(x: torch.Tensor, weight: torch.Tensor):
    return F.conv2d(x, weight, bias=None, stride=1, padding=1)


def op_fft(a: torch.Tensor):
    return torch.fft.fft2(a)


def op_svd(a: torch.Tensor):
    return torch.linalg.svd(a, full_matrices=False)


def compile_jit(op, example_inputs):
    return torch.jit.trace(op, example_inputs, check_trace=False)


def maybe_warmup(device: str, op, *inputs):
    if device in {"cuda", "xpu", "mps"}:
        op(*inputs)
        sync(device)


def run_benchmark(device: str, op, *inputs) -> int:
    sync(device)
    maybe_warmup(device, op, *inputs)
    count = 0
    end = time.perf_counter() + DURATION
    while time.perf_counter() < end:
        op(*inputs)
        sync(device)
        count += 1
    return count


def check_devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    if torch.backends.mps.is_available():
        devices.append("mps")
        mps_mem = _format_bytes(torch.mps.recommended_max_memory())
        print(f"MPS device: Apple Metal Performance Shaders (recommended max memory: {mps_mem})")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        devices.append("xpu")
        print(f"XPU device: {torch.xpu.get_device_name(0)}")
    if devices == ["cpu"]:
        print("WARNING: No accelerator available, only CPU will be benchmarked.")
    print(f"PyTorch: {torch.__version__}")
    print(f"Devices: {devices}\n")
    return devices


def sync(device: str):
    if device == "cuda":
        torch.cuda.synchronize()
    if device == "mps":
        torch.mps.synchronize()
    if device == "xpu":
        torch.xpu.synchronize()


def bench_matmul(device: str) -> int:
    with torch.no_grad():
        a = torch.randn(BATCH_SIZE, MATMUL_SIZE, MATMUL_SIZE, device=device, dtype=torch.float32)
        b = torch.randn(BATCH_SIZE, MATMUL_SIZE, MATMUL_SIZE, device=device, dtype=torch.float32)
        op = compile_jit(op_matmul, (a, b))
        return run_benchmark(device, op, a, b)


def bench_elementwise(device: str) -> int:
    with torch.no_grad():
        a = torch.randn(BATCH_SIZE, ELEMENTWISE_SIZE, ELEMENTWISE_SIZE, device=device, dtype=torch.float32)
        b = torch.randn(BATCH_SIZE, ELEMENTWISE_SIZE, ELEMENTWISE_SIZE, device=device, dtype=torch.float32)
        op = compile_jit(op_elementwise, (a, b))
        return run_benchmark(device, op, a, b)


def bench_reduction(device: str) -> int:
    with torch.no_grad():
        a = torch.randn(BATCH_SIZE, REDUCTION_SIZE, REDUCTION_SIZE, device=device, dtype=torch.float32)
        op = compile_jit(op_reduction, (a,))
        return run_benchmark(device, op, a)


def bench_conv2d(device: str) -> int:
    with torch.no_grad():
        x = torch.randn(
            BATCH_SIZE,
            CONV_CHANNELS,
            CONV_SPATIAL,
            CONV_SPATIAL,
            device=device,
            dtype=torch.float32,
        )
        weight = torch.randn(
            CONV_CHANNELS * 2,
            CONV_CHANNELS,
            3,
            3,
            device=device,
            dtype=torch.float32,
        )
        op = compile_jit(op_conv2d, (x, weight))
        return run_benchmark(device, op, x, weight)


def bench_fft(device: str) -> int:
    with torch.no_grad():
        a = torch.randn(BATCH_SIZE, FFT_SIZE, FFT_SIZE, device=device, dtype=torch.float32)
        op = compile_jit(op_fft, (a,))
        return run_benchmark(device, op, a)


def bench_svd(device: str) -> int:
    with torch.no_grad():
        a = torch.randn(BATCH_SIZE, SVD_SIZE, SVD_SIZE, device=device, dtype=torch.float32)
        op = compile_jit(op_svd, (a,))
        return run_benchmark(device, op, a)


BENCHMARKS = [
    (f"BMM {BATCH_SIZE}x{MATMUL_SIZE}x{MATMUL_SIZE}", bench_matmul),
    (f"Elementwise {BATCH_SIZE}x{ELEMENTWISE_SIZE}²", bench_elementwise),
    (f"Reduction {BATCH_SIZE}x{REDUCTION_SIZE}²", bench_reduction),
    (
        f"Conv2d {CONV_CHANNELS}->{CONV_CHANNELS * 2} ({BATCH_SIZE}x{CONV_SPATIAL}²)",
        bench_conv2d,
    ),
    (f"FFT2 {BATCH_SIZE}x{FFT_SIZE}x{FFT_SIZE}", bench_fft),
    (f"SVD {BATCH_SIZE}x{SVD_SIZE}x{SVD_SIZE}", bench_svd),
]


def main():
    devices = check_devices()
    results = {}  # {bench_name: {device: count}}

    for name, fn in BENCHMARKS:
        results[name] = {}
        for dev in devices:
            print(f"  [{dev.upper():>3}] {name} ...", end="", flush=True)
            try:
                iters = fn(dev)
            except Exception as e:
                print(f" SKIPPED ({e})")
                if dev == "mps":
                    print("       hint: this op may not be implemented or fully optimized on MPS yet.")
                iters = None
                continue
            results[name][dev] = iters
            print(f" {iters} iterations")

    # --- summary table ---
    accelerator_devices = [device for device in devices if device != "cpu"]
    device_columns = [(device.upper(), device) for device in devices]
    speedup_columns = [(f"{device.upper()}x", device) for device in accelerator_devices]
    header = f"{'Benchmark':<28}"
    for title, _ in device_columns:
        header += f" {title:>10}"
    for title, _ in speedup_columns:
        header += f" {title:>10}"
    table_width = max(70, len(header))

    print("\n" + "=" * table_width)
    print(header)
    print("-" * table_width)
    for name, _ in BENCHMARKS:
        row = results.get(name, {})
        cpu_n = row.get("cpu")
        line = f"{name:<28}"
        for _, device in device_columns:
            count = row.get(device)
            count_text = str(count) if count is not None else "N/A"
            line += f" {count_text:>10}"
        for _, device in speedup_columns:
            count = row.get(device)
            speedup = f"{count / cpu_n:.2f}x" if cpu_n and count else "—"
            line += f" {speedup:>10}"
        print(line)
    print("=" * table_width)
    print(f"Each benchmark ran for {DURATION}s. Higher iteration count = faster.\n")


if __name__ == "__main__":
    main()
