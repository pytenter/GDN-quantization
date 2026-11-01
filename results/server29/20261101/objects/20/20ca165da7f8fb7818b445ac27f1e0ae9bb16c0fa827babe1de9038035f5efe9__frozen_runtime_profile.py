"""Launch-only restoration of the recovered, frozen FLA forward profile."""
import hashlib
import json
import os
from pathlib import Path

W=Path(__file__).resolve().parent
CONFIG_DIR=W/'frozen_fla_configs'
CONFIG_SHA256={
 'causal_conv1d_update_kernel.json':'0a3b2436f8c4be23d1aaf425199a0ce7be9d4742ab9c5085f36b9ebbf34eb8cb',
 'chunk_gated_delta_rule_fwd_kernel_h_blockdim64.json':'2915e24bfdc6e311e0de5cfd90289a2d2ee7e581e624a52a67b59dc59d83b8a0',
 'chunk_kda_fwd_kernel_inter_solve_fused.json':'878b44b6cfab77a59b272c436683892d220ad25b837cb482286ac8c08ab016a1',
 'chunk_kda_fwd_kernel_intra_token_parallel.json':'30dec98b4ba66f52647ec43e9e47740d37a0f43453d2785662f9a6259efa3477',
 'chunk_local_cumsum_vector_kernel.json':'42c849c632a928513a4b77ce171af48afaef4f0f8e4a96e8282a8e5020425a27',
 'l2norm_fwd_kernel.json':'82b6b1b151fdd39552fe14a197c4d82dca52e72eeebd26e296a9d56ed1142d70',
}
ENVIRONMENT_KEYS=['PYTHONHASHSEED','CUBLAS_WORKSPACE_CONFIG','LING_GATED_RMSNORM_FIXED_CONFIG','FLA_CACHE_MODE','FLA_CONFIG_DIR']

def validate_files(directory=CONFIG_DIR):
    provenance=json.loads((W/'frozen_runtime_provenance.json').read_text())
    for path,digest in provenance['installed_runtime_source_SHA256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,('FROZEN_FLA_RUNTIME_SOURCE_CHANGED',path)
    directory=Path(directory)
    assert directory.is_dir(),'FROZEN_FLA_CONFIG_DIRECTORY_MISSING'
    assert {p.name for p in directory.iterdir()}==set(CONFIG_SHA256),'FROZEN_FLA_CONFIG_FILE_SET_CHANGED'
    result={}
    for name,digest in CONFIG_SHA256.items():
        p=directory/name
        assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,('FROZEN_FLA_CONFIG_SHA_MISMATCH',name)
        value=json.loads(p.read_text())
        assert value['kernel_name']==name.removesuffix('.json')
        result[value['kernel_name']]=value['default_config']
    return result

def profile_environment():
    validate_files()
    return dict(FLA_CACHE_MODE='default',FLA_CONFIG_DIR=str(CONFIG_DIR))

def validate_environment(env=None):
    env=os.environ if env is None else env
    expected=dict(PYTHONHASHSEED='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',
        LING_GATED_RMSNORM_FIXED_CONFIG='BT16_W8',FLA_CACHE_MODE='default',FLA_CONFIG_DIR=str(CONFIG_DIR))
    for key,value in expected.items():
        assert env.get(key)==value,('FROZEN_FLA_PROFILE_ENV_INVALID',key,env.get(key))
    configurations=validate_files()
    return dict(status='PASS_FROZEN_RUNTIME_PROFILE_INPUTS',environment={k:env.get(k) for k in ENVIRONMENT_KEYS},
                fixed_FLA_config_SHA256=CONFIG_SHA256,default_configurations=configurations)

def config_record(config):
    return dict(kwargs=dict(config.kwargs),num_warps=config.num_warps,
                num_stages=config.num_stages,num_ctas=config.num_ctas)

def selected_config(kernel):
    seen=set()
    while kernel is not None and id(kernel) not in seen:
        seen.add(id(kernel))
        best=getattr(kernel,'best_config',None)
        if best is not None:return config_record(best)
        kernel=getattr(kernel,'fn',None)
    return None

def validate_loaded_profile(require_forward=False):
    inputs=validate_environment()
    import fla.ops.utils.cache as cache
    assert cache.FLA_CACHE_MODE.value=='default','FROZEN_FLA_IMPORT_MODE_MISMATCH'
    assert cache.get_fla_config_dir().resolve()==CONFIG_DIR.resolve(),'FROZEN_FLA_IMPORT_DIRECTORY_MISMATCH'
    loaded={name:cache.load_cached_config(name) for name in inputs['default_configurations']}
    assert loaded==inputs['default_configurations'],'FROZEN_FLA_ACTUAL_LOADER_FALLBACK_OR_MISMATCH'
    observed={}
    if require_forward:
        from fla.modules.conv.triton import kernels
        import fla.modules.fused_norm_gate as norm
        observed['causal_conv1d_update_kernel']=selected_config(kernels.causal_conv1d_update_kernel)
        observed['layer_norm_gated_fwd_kernel']=selected_config(norm.layer_norm_gated_fwd_kernel)
        assert observed['causal_conv1d_update_kernel']==loaded['causal_conv1d_update_kernel'],'FROZEN_CONV_ACTUAL_CONFIG_MISMATCH'
        assert observed['layer_norm_gated_fwd_kernel']==dict(kwargs={'BT':16},num_warps=8,num_stages=3,num_ctas=1),'FROZEN_GATED_NORM_ACTUAL_CONFIG_MISMATCH'
    return dict(status='PASS_ACTUAL_FROZEN_RUNTIME_PROFILE',environment=inputs['environment'],
                FLA_CACHE_MODE_observed=cache.FLA_CACHE_MODE.value,
                FLA_CONFIG_DIR_observed=str(cache.get_fla_config_dir()),
                fixed_FLA_config_SHA256=CONFIG_SHA256,actual_loader_configurations=loaded,
                actually_used_forward_configurations=observed,
                forward_kernels_executed=require_forward,mathematical_sources_modified=False)
