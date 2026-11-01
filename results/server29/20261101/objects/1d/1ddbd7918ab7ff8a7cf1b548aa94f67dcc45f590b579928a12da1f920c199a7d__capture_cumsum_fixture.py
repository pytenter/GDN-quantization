"""Stop at the original line before executing cumsum; no compatibility implementation."""
import inspect, sys, traceback
import closure_common as C

class CapturedBeforeBlockedOperation(Exception): pass

def main():
    import torch
    C.save('provenance/original_parent_stop_identity.json', C.verify_parent())
    C.save('protocol/closure_policy.json',dict(experiment='QWEN_GDN_DETERMINISTIC_CUMSUM_CLOSURE_V1',
        strict=True, engineering_only=True, reference_only_separate_process=True,
        scientific_metrics_started=0, training=0, generation=0, seed=C.SEED,
        implementation_not_created_before_actual_shape=True,
        full_probe_modes={'historical32':'first token prefill + 31 teacher forced cache steps',
                          'dev256':'full 256-token prefill + same fixed teacher tokens in cached decoding'},
        unstable_metadata_exclusions=['wall clock','allocator peaks','process id','serialization container metadata'],
        determinism_objects=['tensor raw bytes','ordered transition tensor hashes'],
        additional_blocker_stop_count=1, no_automatic_retry=True))
    F,core,model,tok,qmod=C.init_model()
    source=C.ARCH.read_text(); lines=source.splitlines()
    blocked=[i+1 for i,s in enumerate(lines) if s.strip()=='g = g.cumsum(dim=-1)']
    C.check(len(blocked)==1,'unique production blocked line')
    C.save('provenance/original_source_identity.json',dict(path=str(C.ARCH),sha256=C.sha(C.ARCH),blocked_line=blocked[0],
        function='torch_chunk_gated_delta_rule',torch_version=torch.__version__,cuda_runtime=torch.version.cuda,
        actual_runtime_function_module=qmod.torch_chunk_gated_delta_rule.__module__))
    row=core.corpus_rows('TRAIN')[0]
    ids=tok(row['raw_text'],add_special_tokens=False).input_ids
    historical=ids[:32]
    # Identity already established in the frozen parent; only revalidate the frozen digest.
    expected=C.read(C.PARENT/'protocol/preregistration_repaired_001.json')['expected_dev_token_sha256']
    C.check(C.tensor_sha(torch.tensor(historical,dtype=torch.long))==expected,'frozen short probe token bytes')
    C.check(len(ids)>=256,'development source length')
    probes=dict(historical32=historical,dev256=ids[:256])
    long_len=min(2048,len(ids)); C.check(long_len>=256,'longer prefill source unavailable')
    probes['benchmark']=ids[:long_len]
    C.save('protocol/frozen_probe_tokens.json',dict(probes=probes,source_split='existing TRAIN development row 0',
        source_raw_text_sha256=row['raw_text_sha256'],source_code_sha256=C.sha(C.READINESS),
        historical_identity_reused=True,original_row_token_count=len(ids),benchmark_length=long_len,
        token_sha256={n:C.tensor_sha(torch.tensor(x,dtype=torch.long)) for n,x in probes.items()}))
    pending={}; captured=[]; original=qmod.torch_chunk_gated_delta_rule
    def observer(query,key,value,*args,**kw):
        pending['chunk_input']=dict(query=C.cpu(query),key=C.cpu(key),value=C.cpu(value),g=C.cpu(kw['g']),beta=C.cpu(kw['beta']),
            chunk_size=kw.get('chunk_size',64),initial_state=C.cpu(kw.get('initial_state')),
            output_final_state=True,use_qk_l2norm_in_kernel=kw.get('use_qk_l2norm_in_kernel',False))
        return original(query,key,value,*args,**kw)
    def tracer(frame,event,arg):
        if event=='line' and frame.f_code.co_filename==str(C.ARCH) and frame.f_lineno==blocked[0]:
            g=frame.f_locals['g']; pending['g']=C.cpu(g)
            pending['audit']=dict(shape=list(g.shape),dtype=str(g.dtype),device=str(g.device),dim=-1,
                final_axis_length=g.shape[-1],chunk_size=frame.f_locals['chunk_size'],
                contiguous=g.is_contiguous(),stride=list(g.stride()),sequence_length=frame.f_locals['sequence_length'],
                captured_immediately_before_original_call=True,torch_strict=torch.are_deterministic_algorithms_enabled())
            raise CapturedBeforeBlockedOperation()
        return tracer
    qmod.torch_chunk_gated_delta_rule=observer
    lengths=[('historical_prefill1',historical[:1]),('historical32',historical),('chunk64',ids[:64]),('chunk128',ids[:128]),('dev256',ids[:256]),('long_prefill',ids[:long_len])]
    try:
        for name,tokens in lengths:
            pending.clear(); sys.settrace(tracer)
            try:
                with torch.inference_mode(): model(input_ids=torch.tensor([tokens],device='cuda:0'),use_cache=True)
                raise RuntimeError('original cumsum not intercepted')
            except CapturedBeforeBlockedOperation: pass
            finally: sys.settrace(None)
            audit=pending['audit']; C.check(audit['final_axis_length']==audit['chunk_size']==64,'unexpected scan width; STOP before implementation')
            fixture=C.pt_save('fixtures/'+name+'.pt',dict(g=pending['g'],chunk=pending['chunk_input']))
            captured.append(dict(name=name,**audit,**fixture,g_raw_sha256=C.tensor_sha(pending['g']),
                                 ids_sha256=C.tensor_sha(torch.tensor(tokens,dtype=torch.long))))
            print('REAL_FIXTURE_CAPTURED '+name+' '+str(audit['shape']),flush=True)
    finally: qmod.torch_chunk_gated_delta_rule=original
    C.save('provenance/cumsum_runtime_shape_audit.json',dict(CUMSUM_RUNTIME_SHAPE_AUDIT='PASS',rows=captured,implementation_present=False))
    C.save('provenance/frozen_cumsum_inputs.json',dict(fixtures=captured,serialized_cpu_copies_of_actual_cuda_inputs=True,
        capture_method='scoped line trace raises before original cumsum; original operations and model unchanged',gate='PASS'))

if __name__=='__main__':
    try: main()
    except Exception as e:
        C.save('provenance/capture.error.json',dict(message=str(e),traceback=traceback.format_exc(),stop=True)); raise
