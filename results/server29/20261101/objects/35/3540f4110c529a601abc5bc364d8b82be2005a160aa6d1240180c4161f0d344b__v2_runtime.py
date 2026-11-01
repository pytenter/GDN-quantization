"""Strict worker-local runtime initialization before any CUDA work."""
import os
import random
import time

def initialize(torch,seed):
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
    assert os.environ.get('LING_GATED_RMSNORM_FIXED_CONFIG')=='BT16_W8'
    assert os.environ.get('PYTHONHASHSEED')==str(seed)
    assert not torch.cuda.is_initialized(),'CUDA_INITIALIZED_BEFORE_STRICT_RUNTIME_INIT'
    torch.use_deterministic_algorithms(True,warn_only=False)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.set_grad_enabled(False)
    random.seed(seed)
    import numpy
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    return snapshot(torch,'before_helpers_and_CUDA',seed)

def snapshot(torch,stage,seed):
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
    return dict(stage=stage,epoch=time.time(),pid=os.getpid(),
        deterministic_algorithms_enabled=torch.are_deterministic_algorithms_enabled(),
        warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cuda_initialized=torch.cuda.is_initialized(),matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark,
        seed=seed,cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'],
        kernel_selector=os.environ['LING_GATED_RMSNORM_FIXED_CONFIG'],torch=torch.__version__,
        cuda_runtime=torch.version.cuda)
