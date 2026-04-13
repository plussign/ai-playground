import torch
import time

@torch.jit.script
def processed_chunk_logic(vec_chunk: torch.Tensor, mat_curr: torch.Tensor, mat_next: torch.Tensor):
    """
    JIT 编译的块内计算逻辑
    计算: sum( V_i @ (M_i @ M_{i+1}) )
    """
    # 在块内执行矩阵合并
    combined = torch.bmm(mat_curr, mat_next)
    combined = torch.bmm(combined, mat_next)
    combined = torch.bmm(combined, mat_next)
    combined = torch.bmm(combined, mat_next)
    combined = torch.bmm(combined, mat_next)

    # 计算向量乘法并求和
    res_sum = torch.bmm(vec_chunk, combined).sum(dim=0)
    return res_sum

def memory_efficient_op(vectors, matrices, chunk_size: int = 1000000):
    """
    分块处理函数，避免显存溢出
    """
    N = matrices.size(0)
    device = matrices.device
    
    # 初始化累加器 (1, 4)
    total_sum = torch.zeros((1, 4), device=device)
    
    # 将 100 万数据分块处理
    # 注意：我们要处理到 N-1，因为存在 i+1
    for i in range(0, N - 1, chunk_size):
        end = min(i + chunk_size, N - 1)
        
        # 提取当前块的数据
        v_chunk = vectors[i:end]
        m_curr = matrices[i:end]
        m_next = matrices[i+1:end+1] # 错位取下一个矩阵
        
        # 调用 JIT 函数计算该块的累加和
        chunk_sum = processed_chunk_logic(v_chunk, m_curr, m_next)
        
        # 累加到全局
        total_sum += chunk_sum
        
    return total_sum / (N - 1)


def synchronize_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()


def warmup_non_cpu(vectors, matrices, device: torch.device, warmup_iters: int = 1):
    """
    非 CPU 设备预热，减少首次运行带来的编译和初始化开销
    """
    if device == "cpu":
        return

    print(f"开始预热 ({device.type}, iterations={warmup_iters})...")
    for _ in range(warmup_iters):
        _ = memory_efficient_op(vectors, matrices, chunk_size=100000)
    synchronize_device(device)

# --- 测试脚本 ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "xpu" if torch.xpu.is_available() else "cpu")
    #device = "cpu"
    N = 100000000
    
    print(f"生成数据中...")
    vecs = torch.randn(N, 1, 4, device=device)
    mats = torch.randn(N, 4, 4, device=device)

    # 非 CPU 设备做完整预热，避免将初始化时间算进正式统计
    warmup_non_cpu(vecs, mats, device)

    print(f"开始分块计算 (Chunk Size=100,000)...")
    if device != "cpu":
        synchronize_device(device)

    start = time.perf_counter()
    
    # 执行分块逻辑
    result = memory_efficient_op(vecs, mats, chunk_size=500000)
    
    if device != "cpu":
        synchronize_device(device)

    print(f"耗时: {time.perf_counter() - start:.6f} 秒")
    print(f"结果:\n{result.view(-1)} device: {result.device}")

if __name__ == "__main__":
    main()