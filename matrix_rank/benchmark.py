import time
import numpy as np
import torch

N = 10_000_000
BATCH_SIZE = 200_000
DURATION = 10

def benchmark_numpy_cpu(matrices, batch_size, duration):

    np.show_config()

    total = 0
    n = len(matrices)
    start = time.time()
    while time.time() - start < duration:
        # take a slice of batch_size
        idx = total % n
        end = idx + batch_size
        if end <= n:
            np.linalg.matrix_rank(matrices[idx:end])
        else:
            np.linalg.matrix_rank(np.concatenate([matrices[idx:], matrices[:end - n]]))
        total += batch_size
    elapsed = time.time() - start
    return total, elapsed


def benchmark_torch_cpu(matrices, batch_size, duration):

    print('PyTorch version:', torch.__version__)
    print()
    print(torch.__config__.show())

    total = 0
    n = len(matrices)
    start = time.time()
    while time.time() - start < duration:
        idx = total % n
        end = idx + batch_size
        if end <= n:
            torch.linalg.matrix_rank(matrices[idx:end])
        else:
            torch.linalg.matrix_rank(torch.cat([matrices[idx:], matrices[:end - n]]))
        total += batch_size
    elapsed = time.time() - start
    return total, elapsed


def benchmark_torch_cuda(matrices, batch_size, duration):
    n = len(matrices)
    # pre-move all data to cuda
    cuda_matrices = matrices.to("cuda")
    # warm up
    torch.linalg.matrix_rank(cuda_matrices[:batch_size])
    torch.cuda.synchronize()

    total = 0
    start = time.time()
    while time.time() - start < duration:
        idx = total % n
        end = idx + batch_size
        if end <= n:
            torch.linalg.matrix_rank(cuda_matrices[idx:end])
        else:
            torch.linalg.matrix_rank(torch.cat([cuda_matrices[idx:], cuda_matrices[:end - n]]))
        torch.cuda.synchronize()
        total += batch_size
    elapsed = time.time() - start
    return total, elapsed


def main():
    batch_size = BATCH_SIZE
    duration = DURATION

    print(f"Creating {N:,} random 8x8 float32 matrices ({N * 64 * 4 / 1e9:.2f} GB)...")
    np_matrices = np.random.randn(N, 8, 8).astype(np.float32)
    print(f"Data ready. Batch size: {batch_size:,}\n")

    # NumPy CPU
    print("Benchmarking NumPy (CPU) — batched...")
    np_total, np_elapsed = benchmark_numpy_cpu(np_matrices, batch_size, duration)
    print(f"  NumPy CPU:  {np_total:,} matrices in {np_elapsed:.2f}s  "
          f"({np_total / np_elapsed:,.0f} matrices/s)\n")

    # PyTorch CPU
    print("Converting to PyTorch tensors for Torch Benchmark...")
    torch_matrices = torch.from_numpy(np_matrices)
    np_matrices = None  # free memory

    print("Benchmarking PyTorch (CPU) — batched...")
    pt_total, pt_elapsed = benchmark_torch_cpu(torch_matrices, batch_size, duration)
    print(f"  PyTorch CPU: {pt_total:,} matrices in {pt_elapsed:.2f}s  "
          f"({pt_total / pt_elapsed:,.0f} matrices/s)\n")

    # PyTorch CUDA
    if torch.cuda.is_available():
        print(f"Benchmarking PyTorch (CUDA) on {torch.cuda.get_device_name(0)} — batched...")
        pt_cuda_total, pt_cuda_elapsed = benchmark_torch_cuda(torch_matrices, batch_size, duration)
        print(f"  PyTorch CUDA: {pt_cuda_total:,} matrices in {pt_cuda_elapsed:.2f}s  "
              f"({pt_cuda_total / pt_cuda_elapsed:,.0f} matrices/s)\n")
    else:
        print("CUDA not available, skipping PyTorch CUDA benchmark.\n")

    print("Done.")


if __name__ == "__main__":
    main()
