import time
import numpy as np
import torch

N = 3_000_000
BATCH_SIZE = 1_000_000


def benchmark_numpy_cpu(matrices, batch_size):
    n = len(matrices)
    all_ranks = []
    processed = 0
    start = time.time()
    while processed < n:
        end = min(processed + batch_size, n)
        ranks = np.linalg.matrix_rank(matrices[processed:end])
        all_ranks.append(ranks)
        processed = end
    elapsed = time.time() - start
    rank_counts = np.bincount(np.concatenate(all_ranks), minlength=9)[:9]
    return elapsed, rank_counts


def benchmark_torch_cpu(matrices, batch_size):
    n = len(matrices)
    all_ranks = []
    processed = 0
    start = time.time()
    while processed < n:
        end = min(processed + batch_size, n)
        ranks = torch.linalg.matrix_rank(matrices[processed:end])
        all_ranks.append(ranks)
        processed = end
    elapsed = time.time() - start
    rank_counts = torch.bincount(torch.cat(all_ranks), minlength=9)[:9]
    return elapsed, rank_counts


def benchmark_torch_cuda(matrices, batch_size):
    n = len(matrices)
    cuda_matrices = matrices.to("cuda")
    # warm up
    torch.linalg.matrix_rank(cuda_matrices[:min(batch_size, n)])
    torch.cuda.synchronize()

    all_ranks = []
    processed = 0
    start = time.time()
    while processed < n:
        end = min(processed + batch_size, n)
        ranks = torch.linalg.matrix_rank(cuda_matrices[processed:end])
        all_ranks.append(ranks.cpu())
        torch.cuda.synchronize()
        processed = end
    elapsed = time.time() - start
    rank_counts = torch.bincount(torch.cat(all_ranks), minlength=9)[:9]
    return elapsed, rank_counts


def print_rank_counts(label, rank_counts):
    print(f"  {label} rank distribution:")
    for r in range(9):
        print(f"    rank {r}: {rank_counts[r]:>10,}")
    print()


def main():
    batch_size = BATCH_SIZE

    print(f"Creating {N:,} random 8x8 float32 matrices ({N * 64 * 4 / 1e9:.2f} GB)...")
    np_matrices = np.random.randn(N, 8, 8).astype(np.float32)
    print(f"Data ready. Batch size: {batch_size:,}\n")

    # NumPy CPU
    print("Benchmarking NumPy (CPU)...")
    np_elapsed, np_ranks = benchmark_numpy_cpu(np_matrices, batch_size)
    print(f"  NumPy CPU:  {np_elapsed:.2f}s  ({N / np_elapsed:,.0f} matrices/s)")
    print_rank_counts("NumPy CPU", np_ranks)

    # PyTorch CPU
    print("Converting to PyTorch tensors...")
    torch_matrices = torch.from_numpy(np_matrices)
    np_matrices = None  # free memory

    print("Benchmarking PyTorch (CPU)...")
    pt_elapsed, pt_ranks = benchmark_torch_cpu(torch_matrices, batch_size)
    print(f"  PyTorch CPU: {pt_elapsed:.2f}s  ({N / pt_elapsed:,.0f} matrices/s)")
    print_rank_counts("PyTorch CPU", pt_ranks)

    # PyTorch CUDA
    if torch.cuda.is_available():
        print(f"Benchmarking PyTorch (CUDA) on {torch.cuda.get_device_name(0)}...")
        pt_cuda_elapsed, pt_cuda_ranks = benchmark_torch_cuda(torch_matrices, batch_size)
        print(f"  PyTorch CUDA: {pt_cuda_elapsed:.2f}s  ({N / pt_cuda_elapsed:,.0f} matrices/s)")
        print_rank_counts("PyTorch CUDA", pt_cuda_ranks)
    else:
        print("CUDA not available, skipping PyTorch CUDA benchmark.\n")

    # Consistency check
    print("=" * 40)
    print("Consistency check:")
    pt_ranks_np = pt_ranks.numpy() if isinstance(pt_ranks, torch.Tensor) else pt_ranks
    np_ranks_np = np_ranks if isinstance(np_ranks, np.ndarray) else np_ranks.numpy()

    match = np.array_equal(np_ranks_np, pt_ranks_np)
    print(f"  NumPy CPU vs PyTorch CPU: {'MATCH' if match else 'MISMATCH'}")
    if torch.cuda.is_available():
        pt_cuda_ranks_np = pt_cuda_ranks.numpy() if isinstance(pt_cuda_ranks, torch.Tensor) else pt_cuda_ranks
        match_cuda = np.array_equal(np_ranks_np, pt_cuda_ranks_np)
        print(f"  NumPy CPU vs PyTorch CUDA: {'MATCH' if match_cuda else 'MISMATCH'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
