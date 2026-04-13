import time
import numpy as np
import torch

N = 3_000_000
BATCH_SIZE = 1_000_000


def benchmark_numpy_cpu(matrices: np.ndarray, batch_size: int):
    n = len(matrices)
    mat_dim = matrices.shape[1] + 1
    all_ranks = []
    processed = 0
    start = time.time()
    while processed < n:
        end = min(processed + batch_size, n)
        ranks = np.linalg.matrix_rank(matrices[processed:end])
        all_ranks.append(ranks)
        processed = end
    elapsed = time.time() - start
    rank_counts = np.bincount(np.concatenate(all_ranks), minlength=mat_dim)[:mat_dim]
    return elapsed, rank_counts

@torch.jit.script
def benchmark_torch_calc(m1 : torch.Tensor, m2: torch.Tensor, batch_size: int, total :int):
    processed = 0

    sumMatrices = torch.zeros_like(m1) 
    row_norms_buffer = torch.empty((batch_size, m1.size(1), 1), dtype=m1.dtype, device=m1.device)

    while processed < total:
        end = min(processed + batch_size, total)

        mat1 = m1[processed:end]
        mat2 = m2[processed:end]

        #stacked_batch = torch.stack((mat1, mat2), dim=0)  # shape (2, batch_size, 8, 8)
        #sumMatrices = stacked_batch.sum(dim=0)  # shape (batch_size, 8, 8)

        sumView = sumMatrices[processed:end]
        sumView.copy_(mat1)
        sumView.add_(mat2)  # alternative way to sum without stacking
        row_norms = row_norms_buffer[:end - processed]

        for _ in range(50):
            torch.baddbmm(sumView, mat1, mat2, beta=1, alpha=1, out=sumView)
            torch.norm(sumView, p=2, dim=2, keepdim=True, out=row_norms)
            row_norms.clamp_min_(1e-12)
            sumView.div_(row_norms)
        
        #ranks = torch.linalg.matrix_rank(sumMatrices)
        #all_ranks[processed:end] = ranks
        processed = end

    # if m1.device.type == "cuda":
    #     torch.cuda.synchronize()
    # elif m1.device.type == "xpu":
    #     torch.xpu.synchronize()

    return sumMatrices

def benchmark_torch_cpu(matrices: torch.Tensor, matrices2: torch.Tensor,batch_size: int):

    n = len(matrices)
    mat_dim = matrices.shape[1] + 1

    all_ranks = [torch.empty(0, dtype=torch.int64)]  # list to hold rank tensors for each batch

    start = time.time()
    res = benchmark_torch_calc(matrices, matrices2, batch_size, n)
    elapsed = time.time() - start

    rank_counts = torch.bincount(torch.cat(all_ranks), minlength=mat_dim)[:mat_dim]

    res = torch.sum(res, dim=(0, 1))
    print(f"Sum of all elements in result: {res.shape}")
    print(f"Sum of all elements in result: {res.sum().item():.4f}")

    return elapsed, rank_counts


def benchmark_torch_cuda(matrices : torch.Tensor, matrices2: torch.Tensor, batch_size: int):

    n = len(matrices)
    mat_dim = matrices.shape[1] + 1

    cuda_matrices = matrices.to("cuda")
    cuda_matrices2 = matrices2.to("cuda")

    # warm up
    benchmark_torch_calc(cuda_matrices, cuda_matrices2, batch_size, batch_size)
    torch.cuda.synchronize()

    all_ranks = torch.zeros(0, dtype=torch.int64, device="cuda")

    start = time.time()
    res = benchmark_torch_calc(cuda_matrices, cuda_matrices2, batch_size, n)
    torch.cuda.synchronize()
    elapsed = max(0.0001, time.time() - start)

    rank_counts = torch.bincount(all_ranks.cpu(), minlength=mat_dim)[:mat_dim]

    res = torch.sum(res, dim=(0, 1))
    print(f"Sum of all elements in result: {res.shape}")
    print(f"Sum of all elements in result: {res.sum().item():.4f}")

    return elapsed, rank_counts

def benchmark_torch_xpu(matrices: torch.Tensor, matrices2: torch.Tensor, batch_size: int):

    n = len(matrices)
    mat_dim = matrices.shape[1] + 1

    xpu_matrices = matrices.to("xpu")
    xpu_matrices2 = matrices2.to("xpu")

    # warm up
    benchmark_torch_calc(xpu_matrices, xpu_matrices2, batch_size, batch_size)
    torch.xpu.synchronize()

    all_ranks = torch.zeros(0, dtype=torch.int64, device="xpu")

    start = time.time()
    res = benchmark_torch_calc(xpu_matrices, xpu_matrices2, batch_size, n)
    elapsed = max(0.0001, time.time() - start)
    torch.xpu.synchronize()

    rank_counts = torch.bincount(all_ranks.cpu(), minlength=mat_dim)[:mat_dim]

    res = torch.sum(res, dim=(0, 1))
    print(f"Sum of all elements in result: {res.shape}")
    print(f"Sum of all elements in result: {res.sum().item():.4f}")

    return elapsed, rank_counts


def print_rank_counts(label, rank_counts):
    print(f"  {label} rank distribution:")
    for r in range(9):
        print(f"    rank {r}: {rank_counts[r]:>10,}")
    print()


def main():
    batch_size = BATCH_SIZE

    print(f"Creating {N:,} random 8x8 float32 matrices ({N * 64 * 4 * 2 / 1e9:.2f} GB)...")
    np_matrices = np.random.randn(N, 8, 8).astype(np.float32)
    np_matrices2 = np.random.randn(N, 8, 8).astype(np.float32)
    print(f"Data ready. Batch size: {batch_size:,}\n")

    # NumPy CPU
    #print("Benchmarking NumPy (CPU)...")
    #np_elapsed, np_ranks = benchmark_numpy_cpu(np_matrices, batch_size)
    #print(f"  NumPy CPU:  {np_elapsed:.2f}s  ({N / np_elapsed:,.0f} matrices/s)")
    #print_rank_counts("NumPy CPU", np_ranks)

    # PyTorch CPU
    print("Converting to PyTorch tensors...")
    torch_matrices = torch.from_numpy(np_matrices)
    torch_matrices2 = torch.from_numpy(np_matrices2)
    np_matrices = None  # free memory
    np_matrices2 = None  # free memory

    print("Benchmarking PyTorch (CPU)...")
    pt_elapsed, pt_ranks = benchmark_torch_cpu(torch_matrices, torch_matrices2, batch_size)
    print(f"  PyTorch CPU: {pt_elapsed:.2f}s  ({N / pt_elapsed:,.0f} matrices/s)")
    #print_rank_counts("PyTorch CPU", pt_ranks)

    # PyTorch CUDA
    if torch.cuda.is_available():
        print(f"Benchmarking PyTorch (CUDA) on {torch.cuda.get_device_name(0)}...")
        pt_cuda_elapsed, pt_cuda_ranks = benchmark_torch_cuda(torch_matrices, torch_matrices2, batch_size)
        print(f"  PyTorch CUDA: {pt_cuda_elapsed:.2f}s  ({N / pt_cuda_elapsed:,.0f} matrices/s)")
        #print_rank_counts("PyTorch CUDA", pt_cuda_ranks)
    else:
        print("CUDA not available, skipping PyTorch CUDA benchmark.\n")

    # PyTorch XPU
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        print(f"Benchmarking PyTorch (XPU) on {torch.xpu.get_device_name(0)}...")
        pt_xpu_elapsed, pt_xpu_ranks = benchmark_torch_xpu(torch_matrices, torch_matrices2, batch_size)
        print(f"  PyTorch XPU: {pt_xpu_elapsed:.2f}s  ({N / pt_xpu_elapsed:,.0f} matrices/s)")
        #print_rank_counts("PyTorch XPU", pt_xpu_ranks)
    else:
        print("XPU not available, skipping PyTorch XPU benchmark.\n")

    # Consistency check
    # print("=" * 40)
    # print("Consistency check:")
    # pt_ranks_np = pt_ranks.numpy() if isinstance(pt_ranks, torch.Tensor) else pt_ranks
    # np_ranks_np = np_ranks if isinstance(np_ranks, np.ndarray) else np_ranks.numpy()

    # match = np.array_equal(np_ranks_np, pt_ranks_np)
    # print(f"  NumPy CPU vs PyTorch CPU: {'MATCH' if match else 'MISMATCH'}")
    # if torch.cuda.is_available():
    #     pt_cuda_ranks_np = pt_cuda_ranks.numpy() if isinstance(pt_cuda_ranks, torch.Tensor) else pt_cuda_ranks
    #     match_cuda = np.array_equal(np_ranks_np, pt_cuda_ranks_np)
    #     print(f"  NumPy CPU vs PyTorch CUDA: {'MATCH' if match_cuda else 'MISMATCH'}")
    # if hasattr(torch, 'xpu') and torch.xpu.is_available():
    #     pt_xpu_ranks_np = pt_xpu_ranks.numpy() if isinstance(pt_xpu_ranks, torch.Tensor) else pt_xpu_ranks
    #     match_xpu = np.array_equal(np_ranks_np, pt_xpu_ranks_np)
    #     print(f"  NumPy CPU vs PyTorch XPU: {'MATCH' if match_xpu else 'MISMATCH'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
