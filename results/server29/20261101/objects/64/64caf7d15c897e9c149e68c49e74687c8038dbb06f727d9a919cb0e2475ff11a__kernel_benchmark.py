"""Non-AIME packed readout CUDA-event/host-dispatch and boundary timing."""
import argparse,json,math,statistics,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from pace_buffer import PACEBuffer,native_l2norm
from pace_quantizer import PackedAffineGroups
from lq_unroll_buffer import LQUnrollBuffer,DeltaRecord
from packed_direct_read import packed_readout


def inputs():
    q=torch.randn((1,16,128),device='cuda',dtype=torch.bfloat16)*.1
    k=torch.randn_like(q)*.1
    v=torch.randn((1,32,128),device='cuda',dtype=torch.bfloat16)*.1
    beta=torch.full((1,32),.3,device='cuda',dtype=torch.bfloat16)
    g=torch.full((1,32),-.01,device='cuda',dtype=torch.float32)
    return q,k,v,beta,g


def run(output):
    torch.set_num_threads(24);torch.set_num_interop_threads(24)
    torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(False)
    torch.manual_seed(20261027)
    mask=next(iter(json.loads((ROOT/'calibration/head_mask.json').read_text())['layers'].values()))['int8_heads']
    mask=[i in mask for i in range(32)]
    rows=[]
    for period in (8,16):
        b=LQUnrollBuffer(period,mask);b.set_prompt_length(0)
        for i in range(period-1):b.step(*inputs())
        q,k,v,beta,g=inputs()
        encoded=PackedAffineGroups.from_tensor(k.reshape(1,16,4,32),4)
        key=b._token_keys(DeltaRecord(encoded,None,g))
        query=native_l2norm(q.repeat_interleave(2,1)).float()/math.sqrt(128)
        records=b._slots[:b.pending_tokens]
        pk=torch.stack([b._token_keys(t) for t in records]).contiguous()
        pu=torch.stack([t.u for t in records]).contiguous()
        alphas=torch.stack([t.log_decay for t in records]+[g]).exp().contiguous()
        def call():return packed_readout(b.packed_state,pk,pu,alphas,query,key,v,beta,b._head_maps)
        for _ in range(10):call()
        torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):captured=call()
        times=[];host=[]
        for repeat in range(10):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(50):graph.replay()
            end.record();end.synchronize();times.append(start.elapsed_time(end)*1000/50)
            torch.cuda.synchronize();t=time.perf_counter();last=call();host.append((time.perf_counter()-t)*1e6);torch.cuda.synchronize()
        torch.cuda.synchronize();t=time.perf_counter();b.step(q,k,v,beta,g);torch.cuda.synchronize()
        boundary_us=(time.perf_counter()-t)*1e6
        assert b.decode_dense_materializations==1 and b.pending_tokens==0
        rows.append(dict(period=period,pending_before_readout=period-1,
            fused_packed_kernels_per_readout=2,CUDA_event_graph_readout_median_us=statistics.median(times),
            CUDA_event_graph_readout_all_us=times,ordinary_host_dispatch_median_us=statistics.median(host),
            host_dispatch_all_us=host,boundary_step_host_synchronized_us=boundary_us,
            boundary_step_includes_K4_codec_and_validation_and_original_encoder=True,
            state_materializations_at_boundary=b.decode_dense_materializations,
            microbenchmark_batch_size=1,head_count=32,K=128,V=128,
            base_bytes=b.packed_state.storage_nbytes(),buffer_capacity_bytes=b.storage_report()['decode_buffer_capacity_nbytes'],
            vector_scratch_peak_bytes=b.peak_vector_scratch_bytes))
        print(json.dumps(dict(phase='KERNEL_BENCH_DONE',period=period,readout_us=statistics.median(times))),flush=True)
    result=dict(status='COMPLETE_PACKED_KERNEL_MICROBENCH',rows=rows,
        timing_scope='CUDA events over graph replays of the two packed kernels; excludes K4 codec, normalization and vector-stack preparation. Ordinary Python dispatch is separately timed. Boundary is a single synchronized full buffer.step observation, not a stable median.',
        pure_SM_activity_time_measured=False,uses_fused_GPU_kernel=True,model_loads=0,new_AIME_samples=0)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:json.dump(result,f,indent=2)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);run(p.parse_args().output)
