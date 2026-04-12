import time
import importlib
import numpy as np
import torch


def print_torch_devices():
    print('PyTorch devices:')

    # CPU is always available for PyTorch.
    print('- CPU: available')

    cuda_available = torch.cuda.is_available()
    print(f'- CUDA available: {cuda_available}')
    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f'- CUDA device count: {device_count}')
        for idx in range(device_count):
            props = torch.cuda.get_device_properties(idx)
            capability = f'{props.major}.{props.minor}'
            total_mem_gb = props.total_memory / (1024 ** 3)
            print(f'  - cuda:{idx}')
            print(f'    name: {props.name}')
            print(f'    capability: {capability}')
            print(f'    multiprocessors: {props.multi_processor_count}')
            print(f'    total_memory_gb: {total_mem_gb:.2f}')
    else:
        print('- CUDA device count: 0')

    # MPS is mainly available on Apple Silicon/macOS builds.
    mps_backend = getattr(torch.backends, 'mps', None)
    mps_available = bool(mps_backend and mps_backend.is_available())
    mps_built = bool(mps_backend and mps_backend.is_built())
    print(f'- MPS built: {mps_built}, available: {mps_available}')

    # XPU is available on some Intel-enabled builds.
    xpu = getattr(torch, 'xpu', None)
    xpu_available = bool(xpu and xpu.is_available())
    print(f'- XPU available: {xpu_available}')
    if xpu_available:
        xpu_count = xpu.device_count()
        print(f'- XPU device count: {xpu_count}')
        for idx in range(xpu_count):
            print(f'  - xpu:{idx}')
            name_fn = getattr(xpu, 'get_device_name', None)
            if callable(name_fn):
                print(f'    name: {name_fn(idx)}')
    else:
        print('- XPU device count: 0')


def print_directml_devices():
    print('DirectML (torch-directml):')

    if importlib.util.find_spec('torch_directml') is None:
        print('- installed: False')
        print('- note: install with `pip install torch-directml` on Windows for AMD/Intel GPU support')
        return

    print('- installed: True')
    try:
        import torch_directml  # type: ignore[import-not-found]

        dml_device = torch_directml.device()
        print(f'- device: {dml_device}')

        # Small allocation test to verify runtime availability.
        test_tensor = torch.tensor([1.0, 2.0, 3.0], device=dml_device)
        print(f'- runtime_test: ok, tensor_device={test_tensor.device}, sum={test_tensor.sum().item()}')
    except Exception as ex:
        print(f'- runtime_test: failed ({type(ex).__name__}: {ex})')

def main():
    print('NumPy version:', np.__version__)
    print()
    print('NumPy config:')
    np.show_config()
    print()
    print('PyTorch version:', torch.__version__)
    print()
    print(torch.__config__.show())
    print()
    print_torch_devices()
    print()
    print_directml_devices()

if __name__ == "__main__":
    main()