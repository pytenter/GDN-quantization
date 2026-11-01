"""Strict CPU audit of actual unit outputs and release; no scoring/GPU calls.

The original 52 per-unit mathematical/native checks are shared unchanged.
Only the finite fixed-origin48/current4 evidence routing is new.
"""
from pathlib import Path
import execution_control as C
from control_plan import build_qualification_plan

def inspect_unit(root,unit,*,source_sha256,permit_sha256,triton_manifest_sha256,triton_task_root):
    root=Path(root)
    t=root/'execution'/unit['id'];w=C.read(t/'RESULT.json');i=C.read(t/'RESOURCE_INTERVAL.json');d=C.read(t/'DISPATCH.json');start=C.read(t/'STARTED.json')
    copy=dict(w);payload=copy.pop('payload_sha256')
    C.require(C.objsha(copy)==payload and w['status']=='PASS' and w['unit']==unit['id'],'Actual qualification payload mismatch')
    C.require(w['identity']==i['identity']==start['identity'] and w['gpu']==unit['gpu'] and w['gpu_uuid']==unit['gpu_uuid'],'Actual qualification identity/UUID differs')
    C.require(i['exit_returncode']==0 and i['release']=='PASS','Actual exit0/release missing')
    C.require(w['source_freeze_sha256']==source_sha256 and w['permit_sha256']==permit_sha256==d['permit_sha256'],'Current execution source/permit differs')
    driver=C.read(t/'DRIVER_IDENTITY.json');post=C.read(t/'POSTFLIGHT_LIBRARY.json')
    C.require(driver['status']=='PASS' and driver['actual_uuid']==unit['gpu_uuid'] and driver['visible_device_count']==1 and post['status']=='PASS','Actual driver/native closure failed')
    r=w['result'];kind=unit['kind']
    triton=C.read(t/'TRITON_DEPENDENCY_ADMISSION.json')
    C.require(triton['status']=='PASS_FRESH_TRITON_DEPENDENCIES_ACTUAL_IMPORT' and
              r['triton_dependency_admission_sha256']==C.sha(t/'TRITON_DEPENDENCY_ADMISSION.json') and
              triton['manifest_sha256']==triton_manifest_sha256 and
              triton['task_root']==triton_task_root and
              triton['actual_triton_version']==triton['version']=='3.1.0',
              'Actual new Triton compiler/source/native admission missing')
    if kind=='codec_fields':
        C.require(r['status']=='PASS' and r['actual_cuda'] is True and r['independent_reference_pass'] is True,'Actual GPU fields/independent oracle missing')
        if unit['gpu']==1:
            C.require(r['non_aime_diagnostics']['status']=='PASS_ACTUAL_NONAIME_DIAGNOSTICS_NO_PROXY_THRESHOLD',
                      'Actual fixed nonAIME diagnostics missing; no metric quality threshold applies')
    else:
        C.require(C.read(t/'PREFLIGHT_LIBRARY.json')['status']=='PASS','Actual preflight nativeclosure failed')
    if kind=='native_persistent_fresh':
        a=r['audit'];n=unit['tokens']
        C.require(r['tokens']==512 and a['qdq_layer_updates']==24*n and a['next_consumption_checks']==24*(n-1),'Real32/128/512 all-layer QDQ/next count differs')
        C.require(len(r['writebacks_by_layer'])==24 and all(v==n for v in r['writebacks_by_layer'].values()),'Actual512 writeback layers incomplete')
        trace=C.read(t/'TRACE.json');C.require(len(trace['tokens'])==n and [x['token'] for x in trace['tokens']]==list(range(n)) and trace['status']==('REFERENCE_COLLECTED' if unit['repeat']==0 else 'PASS_FULL_ACTUAL_TENSOR_BYTES'),'Full fresh actual byte trace missing')
    elif kind=='nonAIME_free_generation_fresh':
        n=r['generated_token_count'];p=r['input_tokens'];total=p+n-1;a=r['quantizer_audit']
        C.require(0<n<=64 and r['actual_model_forward_calls']==total and a['qdq_layer_updates']==24*total and r['sampler_calls']==n,'Actual sampler/prompt/fresh canary count differs')
        C.require(r['prompt_layer_QDQ']==24*p and r['decode_layer_QDQ']==24*(n-1),'Prompt token0/decode QDQ missing')
        C.require(a['next_consumption_checks']==24*(total-1),'Full short actual next-consumption missing')
        ts=r['generated_token_ids'];eos=ts[-1] in r['stop_token_ids']
        C.require(len(ts)==n and r['finish_reason']==('eos' if eos else 'length') and (eos or n==64),'Actual EOS/length invalid')
        if unit['repeat']==1:
            prior=C.read(root/'execution'/unit['reference_unit']/'RESULT.json')['result']
            for key in ('generated_token_ids','input_ids_sha256','initial_generator_sha256','final_generator_sha256','finish_reason'):
                C.require(prior[key]==r[key],'Two fresh original sampler/RNG payloads differ:'+key)
            C.require(C.read(t/'TRACE.json')['status']=='PASS_FULL_ACTUAL_TENSOR_BYTES','Actual sampler full-byte check missing')
    elif kind=='real_model_memory_pressure':
        C.require(r['status']=='PASS' and {x['condition'] for x in r['cases']}=={'W','R','RW'},'Actual three-codec pressure missing')
        for x in r['cases']:
            C.require(x['total_context']>=82293 and x['actual_short_QDQ_tokens']==16 and x['free_bytes']>=1024**3,'Maximum-context actual shortQDQ/nativeSDPA resource gate failed')
            C.require(x['pressure_scope']=='SYNTHETIC_KV_NATIVE_SDPA_WITH_LIVE_MODEL_NOT81K_GDN_TRAJECTORY','Pressure scope claim changed')
    elif kind=='matched_performance':
        C.require(r['tokens']==4096 and r['timing']['short']['TPS']>0 and r['timing']['late']['TPS']>0,'Matched actual early/late TPS missing')
        for window in ('short','late'):
            C.require(r['timing'][window]['measured']==128 and r['timing'][window]['observer'] is False and r['timing'][window]['sampling_invoked'] is True,'Measurement window observer/sampler differs')
        q=r['profiler'];C.require(q['status']=='PASS_ACTUAL16TOKEN_CUDA_PROFILE' and q['actual_CUDA_kernel_calls']>0 and q['quantizer_CUDA_milliseconds']>0,'Actual quantizer kernel/time profile missing')
    return dict(unit=unit['id'],kind=kind,gpu=unit['gpu'],gpu_uuid=unit['gpu_uuid'],result_sha256=C.sha(t/'RESULT.json'),resource_interval_sha256=C.sha(t/'RESOURCE_INTERVAL.json'),actualexit0=True,release='PASS')

def inspect_qualification(root,plan):
    from qualification_reuse import admit_reused48,remaining_units
    root=Path(root);expected=build_qualification_plan(C.UUIDS,C.AUTHORIZED_GPUS)
    C.require(plan['qualification_units']==expected['qualification_units'],'Finite qualification roster differs')
    reused=admit_reused48(root,plan)
    rows=list(reused['units'])
    for unit in remaining_units(plan):
        row=inspect_unit(root,unit,source_sha256=C.sha(root/'SOURCE_FREEZE.json'),permit_sha256=C.sha(root/'GPU_PERMIT.json'),triton_manifest_sha256=C.sha(root/'analysis/TRITON_RUNTIME_FRESH_SHA.json'),triton_task_root=str(root.resolve()))
        t=root/'execution'/unit['id'];receipt=C.read(t/'DURABLE29_RECEIPT.json')
        C.require(receipt['status']=='PASS_DURABLE29_CLOSED_UNIT' and receipt['unit']==unit['id'] and receipt['formal'] is False and receipt['source_freeze_sha256']==C.sha(root/'SOURCE_FREEZE.json') and receipt['permit_sha256']==C.sha(root/'GPU_PERMIT.json'),'New qualification lacks bound durable29 output receipt')
        row.update(evidence_origin='CURRENT_NEW_EXECUTION',durable29_receipt_sha256=C.sha(t/'DURABLE29_RECEIPT.json'));rows.append(row)
    C.require(len(rows)==52 and len({x['unit'] for x in rows})==52,'Combined exact48+4 qualification roster differs')
    return dict(status='PASS_STRICT48_REUSED_PLUS4_CURRENT_EXIT0_RELEASE',units=rows,count=52,reused_count=48,current_count=4,CPU_mock_credit=0,historical_AIME_result_credit=0,proxy_MSE_or_accuracy_gate=False,reuse_admission_sha256=reused['admission_sha256'])
