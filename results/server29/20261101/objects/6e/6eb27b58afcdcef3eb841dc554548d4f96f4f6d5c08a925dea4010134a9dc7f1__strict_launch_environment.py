"""Future launch candidate only. Does not restart the stopped execution."""
def strict_launch_environment(base, seed, gpu_uuid, physical_index):
    env=dict(base)
    env.update(PYTHONHASHSEED=str(seed),CUBLAS_WORKSPACE_CONFIG=':4096:8',
        LING_GATED_RMSNORM_FIXED_CONFIG='BT16_W8',PYTHONDONTWRITEBYTECODE='1',
        CUDA_VISIBLE_DEVICES=str(physical_index),LING_TARGET_GPU_UUID=gpu_uuid)
    env.pop('LING_EXECUTION_LANE',None)
    return env
