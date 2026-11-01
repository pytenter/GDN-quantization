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

def inspect_current_profile_lifecycle(root,unit):
    # Additional current4 lifecycle gate; original mathematical inspect_unit
    # above is unchanged and remains the sole validator for original reused48.
    root=Path(root);target=root/'execution'/unit['id'];wrapper=C.read(target/'RESULT.json');result=wrapper['result']
    C.require(unit['kind']=='matched_performance' and unit['gpu']==1,'CUPTI gate limited to current four exact matched units')
    paths=['CUPTI_PRELOAD_ADMISSION.json','PROFILE_COLLECTION_COMPLETED.json','CUPTI_PROFILE_DETACH.json','CUPTI_PROFILE_DETACH_EVENTS.jsonl']
    C.require(all((target/n).is_file() and not (target/n).is_symlink() for n in paths),'Current profile lifecycle receipts absent/unsafe')
    preload=C.read(target/paths[0]);collection=C.read(target/paths[1]);detach=C.read(target/paths[2]);manifest=C.cupti_binding(root);elf=manifest['files'][0]
    common=dict(unit=unit['id'],identity=wrapper['identity'],gpu_uuid=unit['gpu_uuid'],source_freeze_sha256=C.sha(root/'SOURCE_FREEZE.json'),permit_sha256=C.sha(root/'GPU_PERMIT.json'),manifest_sha256=C.sha(root/'analysis/CUPTI_LIFECYCLE_BINDING.json'))
    for name,row in [('preload',preload),('collection',collection),('detach',detach)]:
        C.require(all(row.get(k)==v for k,v in common.items()),'Current profile lifecycle exact identity/source/CUPTI binding differs:'+name)
    C.require(preload['status']=='EXACT_EXISTING_CUPTI_ELF_LOADED_BEFORE_DRIVER_INIT_NO_CUPTI_API_CALL' and preload['already_held'] is False and preload['CUDA_Driver_API_calls']==0 and preload['CUPTI_API_calls']==0 and preload['model_forward_calls']==0 and preload['QDQ_calls']==0 and preload['normal_process_teardown_required'] is True,'Actual unique pre-driver CUPTI preload missing')
    for row in (preload,detach):C.require(row['path']==elf['path'] and row['sha256']==elf['sha256'],'CUPTI preload/detach handle differs from exact frozen ELF')
    C.require(collection['status']=='PASS_ACTUAL_PROFILE_COLLECTION_COMPLETED_BEFORE_DETACH' and collection['profile_summary']==result['profiler'] and collection['profile_summary_sha256']==C.objsha(result['profiler']),'Original16-token profile summary changed across cleanup')
    C.require(detach['status']=='DOCUMENTED_CUPTI_FLUSH_AND_DETACH_RETURNED_SUCCESS' and detach['flush_returncode']==detach['finalize_returncode']==0 and detach['CUPTI_API_calls']==2 and detach['profile_summary_sha256']==C.objsha(result['profiler']) and detach['profiler_collection_already_completed'] is True and detach['CUDA_or_CUPTI_APIs_not_disabled'] is True and detach['signal_or_exit_suppression'] is False and detach['actual_process_exit_still_required'] is True and result['profiler_native_lifecycle']==detach,'Actual completed profile cleanup/normal-exit requirement differs')
    import json
    events=[json.loads(line) for line in (target/paths[3]).read_text().splitlines() if line.strip()]
    expected=['CYCLIC_GC_WHILE_CUDA_RUNTIME_ALIVE','CUDA_WORK_SYNCHRONIZED_BEFORE_ACTIVITY_FLUSH','CUPTI_ACTIVITY_FLUSH_BEGIN','CUPTI_ACTIVITY_FLUSH_RETURN','CUPTI_FINALIZE_BEGIN','CUPTI_FINALIZE_RETURN']
    C.require([x['phase'] for x in events]==expected and all(x['pid']==wrapper['identity']['pid'] for x in events) and all(a['monotonic']<=b['monotonic'] for a,b in zip(events,events[1:])),'CUPTI cleanup actual journal order/PID differs')
    C.require(events[2]['flags']==0 and events[3]['returncode']==events[5]['returncode']==0,'CUPTI activity flush/finalize did not actually return0')
    for name in ('PREFLIGHT_LIBRARY.json','POSTFLIGHT_LIBRARY.json'):
        native=C.read(target/name);selected=[x for x in native['selected'] if x['path']==elf['path']]
        C.require(len(selected)==1 and selected[0]['sha256']==elf['sha256'] and selected[0]['bytes']==elf['bytes'] and selected[0]['match'] is True,'Actual mapped CUPTI ELF missing from fresh native closure:'+name)
        segments=[x for x in native['segments'] if x['path']==elf['path']]
        C.require(segments and all(x['mapped_inode']==x['actual_inode']==elf['inode'] for x in segments),'Actual mapped CUPTI inode missing/different:'+name)
    return dict(status='PASS_CURRENT_REAL_PROFILE_PRELOAD_DETACH_AND_NATIVE_BINDING',receipts={name:C.sha(target/name) for name in paths},qualification_credit_from_lifecycle_alone=0)

def inspect_qualification(root,plan):
    from qualification_reuse import admit_reused48,remaining_units
    root=Path(root);expected=build_qualification_plan(C.UUIDS,C.AUTHORIZED_GPUS)
    C.require(plan['qualification_units']==expected['qualification_units'],'Finite qualification roster differs')
    reused=admit_reused48(root,plan)
    rows=list(reused['units'])
    for unit in remaining_units(plan):
        row=inspect_unit(root,unit,source_sha256=C.sha(root/'SOURCE_FREEZE.json'),permit_sha256=C.sha(root/'GPU_PERMIT.json'),triton_manifest_sha256=C.sha(root/'analysis/TRITON_RUNTIME_FRESH_SHA.json'),triton_task_root=str(root.resolve()))
        row['profiler_lifecycle']=inspect_current_profile_lifecycle(root,unit)
        t=root/'execution'/unit['id'];receipt=C.read(t/'DURABLE29_RECEIPT.json')
        C.require(receipt['status']=='PASS_DURABLE29_CLOSED_UNIT' and receipt['unit']==unit['id'] and receipt['formal'] is False and receipt['source_freeze_sha256']==C.sha(root/'SOURCE_FREEZE.json') and receipt['permit_sha256']==C.sha(root/'GPU_PERMIT.json'),'New qualification lacks bound durable29 output receipt')
        row.update(evidence_origin='CURRENT_NEW_EXECUTION',durable29_receipt_sha256=C.sha(t/'DURABLE29_RECEIPT.json'));rows.append(row)
    C.require(len(rows)==52 and len({x['unit'] for x in rows})==52,'Combined exact48+4 qualification roster differs')
    return dict(status='PASS_STRICT48_REUSED_PLUS4_CURRENT_EXIT0_RELEASE',units=rows,count=52,reused_count=48,current_count=4,CPU_mock_credit=0,historical_AIME_result_credit=0,proxy_MSE_or_accuracy_gate=False,reuse_admission_sha256=reused['admission_sha256'])
