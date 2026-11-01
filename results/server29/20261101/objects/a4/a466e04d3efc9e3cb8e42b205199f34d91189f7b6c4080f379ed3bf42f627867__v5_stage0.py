"""CPU-only V4 root cause, real safetensors header census and current NVML inventory."""
import math
import re
import struct
from v5_common import *

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'CPU_ONLY'
    gate=checked(V4/'PHYSICAL_RESOURCE_FEASIBILITY.json')
    identities=checked(V4/'V4_REUSED_SOURCE_IDENTITY.json')
    raw=query(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,driver_version,pci.bus_id','--format=csv,noheader,nounits'])
    contexts=query(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits'])
    import pynvml as nv
    nv.nvmlInit(); GPUs=[]
    try:
        for i in range(nv.nvmlDeviceGetCount()):
            h=nv.nvmlDeviceGetHandleByIndex(i); mem=nv.nvmlDeviceGetMemoryInfo(h)
            name=nv.nvmlDeviceGetName(h); uuid=nv.nvmlDeviceGetUUID(h)
            if isinstance(name,bytes): name=name.decode()
            if isinstance(uuid,bytes): uuid=uuid.decode()
            procs=[dict(pid=p.pid,used_gpu_memory_bytes=p.usedGpuMemory) for p in nv.nvmlDeviceGetComputeRunningProcesses(h)]
            GPUs.append(dict(index=i,uuid=uuid,name=name,total_bytes=mem.total,used_bytes=mem.used,free_bytes=mem.free,
                compute_processes=procs,utilization_gpu=nv.nvmlDeviceGetUtilizationRates(h).gpu,
                candidate_idle=not procs and mem.used<64*1024**2 and nv.nvmlDeviceGetUtilizationRates(h).gpu==0,
                historical_numerical_failure=i==3,numerical_qualification='NOT_RUN_V5'))
    finally: nv.nvmlShutdown()
    import torch
    assert not torch.cuda.is_initialized()
    inventory=dict(epoch=time.time(),GPUs=GPUs,raw_nvidia_smi=raw,raw_compute=contexts,PCIe_topology=query(['nvidia-smi','topo','-m']),
        cpu_meminfo=Path('/proc/meminfo').read_text(),disk=dict(total=shutil.disk_usage(OUT).total,free=shutil.disk_usage(OUT).free),
        runtime=dict(torch=torch.__version__,cuda_runtime=torch.version.cuda,kernel=__import__('platform').release()),
        CPU_only_no_CUDA_context=True)
    emit(OUT/'GPU_AVAILABILITY_AND_TOPOLOGY.json',inventory)
    sf=[Path(p) for p in identities['assets'] if p.endswith('.safetensors')]
    assert len(sf)==32
    weights={}; total=0; header_rows=[]
    sizes={'BF16':2,'F16':2,'F32':4,'F64':8,'I64':8,'I32':4,'I16':2,'I8':1,'U8':1,'BOOL':1}
    for p in sorted(sf):
        with p.open('rb') as f:
            n=struct.unpack('<Q',f.read(8))[0]; header=f.read(n); h=json.loads(header)
        header_rows.append(dict(path=str(p),header_sha256=hashlib.sha256(header).hexdigest(),source=identities['assets'][str(p)]))
        for name,v in h.items():
            if name=='__metadata__': continue
            b=math.prod(v['shape'])*2 # original from_pretrained BF16, frozen weights no precision change
            assert math.prod(v['shape'])*sizes[v['dtype']]==v['data_offsets'][1]-v['data_offsets'][0]
            group=re.match(r'model\.layers\.(\d+)\.',name)
            key='layer'+group[1] if group else name.split('.weight')[0]
            weights[key]=weights.get(key,0)+b; total+=b
    micro=checked(V4/'runs/micro_pg19-test-860_gpu6_fresh0.json')
    last=micro['stream']['hybrid_cache_inventories']['8447']['FP']['layers']
    census=[]
    for l in range(24):
        entries={k:v for k,v in last.items() if k.startswith(f'layer{l}.')}
        fixed=sum(v['bytes'] for k,v in entries.items() if '.conv' in k or v['dtype']=='torch.float32')
        rate=sum(v['bytes'] for k,v in entries.items() if '.conv' not in k and v['dtype']!='torch.float32')/8448
        assert rate in (0,10240)
        census.append(dict(layer=l,kind='KDA' if rate==0 else 'MLA',BF16_weight_bytes=weights['layer'+str(l)],
            recurrent_bytes=sum(v['bytes'] for k,v in entries.items() if v['dtype']=='torch.float32'),
            convolution_bytes=sum(v['bytes'] for k,v in entries.items() if '.conv' in k),
            MLA_KV_bytes_per_token=rate,cache_bytes_per_branch_32K=fixed+rate*32768,actual_V4_cache=entries))
    outside={k:v for k,v in weights.items() if not k.startswith('layer')}
    candidates=[]
    embed=sum(v for k,v in outside.items() if 'word_embeddings' in k); tail=sum(outside.values())-embed
    for boundary in range(1,24):
        sizes32=[embed+sum(x['BF16_weight_bytes']+3*x['cache_bytes_per_branch_32K'] for x in census[:boundary]),
                 tail+sum(x['BF16_weight_bytes']+3*x['cache_bytes_per_branch_32K'] for x in census[boundary:])]
        candidates.append(dict(boundary=boundary,weights_plus_three_native_caches_32K=sizes32,max_bytes=max(sizes32)))
    best=min(candidates,key=lambda x:(x['max_bytes'],x['boundary']))
    # No reserved-allocated fragmentation claim: V4 did not record allocator active/split statistics.
    rows=[]
    for d in gate['documents']:
        v=checked(Path(d['micro_canonical_path'])); rows.append(dict(document=d['document'],canonical_sha256=digest(d['micro_canonical_path']),
            peak_allocated_bytes=d['gpu_memory']['peak_allocated_bytes'],peak_reserved_bytes=d['gpu_memory']['peak_reserved_bytes'],
            V4_projection_bytes=d['conservative_projected_reserved_bytes'],capacity_bytes=d['gpu_memory']['total_bytes'],
            driver_non_torch_overhead_measured_bytes=d['NVML_measured_overhead_bytes'],
            native_cache_growth_bytes_per_token=d['native_bytes_per_token'],native_three_caches_32K_bytes=d['projected_native_32K_bytes'],
            active_bytes=None,inactive_split_bytes=None,allocator_classification='NOT_IDENTIFIABLE_FROM_V4_TELEMETRY',
            workspace_upper_from_peak_minus_weights_and_native_bytes=d['gpu_memory']['peak_allocated_bytes']-total-d['native_8448_bytes']))
    emit(OUT/'V4_PHYSICAL_GATE_ROOT_CAUSE.json',dict(status='ESTABLISHED_PREDICTED_32K_CAPACITY_EXCEEDED',
        first_failed_predicate=gate['errors'][0],classification='PREDICTED_32K_CAPACITY_EXCEEDED',not_actual_OOM=True,
        projection_formula='peak_reserved + 1.5*(future_native_cache_delta + unexplained_growth*(32768/8448)) + measured_NVML_overhead',
        statistical_error_proven=False,allocator_pressure_not_proven=True,rows=rows,source_sha256=digest(V4/'PHYSICAL_RESOURCE_FEASIBILITY.json')))
    emit(OUT/'PER_LAYER_MEMORY_CENSUS.json',dict(status='CPU_HEADER_AND_V4_NATIVE_CACHE_CENSUS',layers=census,outside_layer_weights=outside,
        total_BF16_parameter_bytes=total,weight_headers=header_rows,dual_candidates=candidates,best_dual_boundary=best,
        one_card_weights_plus_three_caches_32K=total+3*sum(x['cache_bytes_per_branch_32K'] for x in census),
        exact_loaded_parameter_storage_and_workspace='TO_BE_MEASURED_V5',all_32_weights_current_metadata_only=True))
    g6=next(x for x in GPUs if x['index']==6)
    assert g6['uuid']=='GPU-de53f243-b271-521c-d875-d96611ed8574' and g6['candidate_idle']
    peers=[x for x in GPUs if x['index'] in (5,4) and x['candidate_idle'] and '3090' in x['name']]
    assert peers, 'NO_IDLE_SAME_PCIE_PEER'
    peer=peers[0]; selected=[g6,peer]
    assert all('3090' in x['name'] and 24*1024**3 <= x['total_bytes'] < 25*1024**3 for x in selected)
    mapping=dict(status='PROPOSED_NOT_NUMERICALLY_QUALIFIED',route='STATIC_CONTIGUOUS_LAYER_MODEL_PARALLEL',
        physical_gpus=[x['index'] for x in selected],physical_gpu_uuids={str(x['index']):x['uuid'] for x in selected},
        boundary=best['boundary'],local_cuda={'embedding_and_early_layers':0,'late_layers_norm_lmhead':1},
        layer_local_device={str(l):int(l>=best['boundary']) for l in range(24)},
        boundary_selection='minimize maximum BF16 parameter + three full native 32K cache bytes, lexicographic smallest boundary tie break',
        candidate_capacity_is_not_qualification=True,estimated_base_per_GPU=best['weights_plus_three_native_caches_32K'],
        pairing='SAME_QUALIFIED_DEVICE_MAP_PAIRING',cache_owned_by_layer_device=True,hidden_boundary_dtype='original BF16',
        input_devices_order=','.join(str(x['index']) for x in selected),no_data_parallel_no_model_duplication=True)
    emit(OUT/'MODEL_PARALLEL_DEVICE_MAP.json',mapping)
    print(json.dumps(dict(best_boundary=best,selected_GPU_indices=mapping['physical_gpus'],total_weights_bytes=total,root_cause=gate['errors'][0])))

if __name__=='__main__': main()
