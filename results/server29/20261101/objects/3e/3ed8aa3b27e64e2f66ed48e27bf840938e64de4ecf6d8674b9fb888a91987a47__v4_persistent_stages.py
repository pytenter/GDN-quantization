"""Original 8448 physical gate, no arbitrary GPUh or VRAM percentage cap."""
import math
import shutil
from pathlib import Path
from v4_persistent_common import *

def physical_micro_gate():
    cfg,sel,freeze=verify(); rows=[]; errors=[]
    for e in sel['documents']:
        v,p=complete('micro',e['document']); s=v['stream']; assert 'results' not in s,'MICRO_OUTCOME_BLINDNESS_FAIL'
        assert s['endpoint']==8448 and s['every_token_qdq_including_after_anchors']
        inv=s['hybrid_cache_inventories']; assert all(str(x) in inv for x in (2047,2048,8191,8192,8447))
        first,last=inv['0'],inv['8447']; b0=sum(x['cache_bytes'] for x in first.values()); bn=sum(x['cache_bytes'] for x in last.values())
        rate=(bn-b0)/8447; assert rate>=0
        projection=bn+(32768-8448)*rate
        for pos,branches in inv.items(): assert sum(x['cache_bytes'] for x in branches.values())==b0+int(pos)*rate,'NATIVE_CACHE_NONLINEAR_GROWTH'
        samples=v['progress_memory_storage']; assert samples[-1]['position']==8448
        growth=[x['allocated_bytes'] for x in samples if x['position']>=2048]
        unexplained=max(0,growth[-1]-growth[0]-(8448-2048)*rate)
        guards=[__import__('json').loads(line) for line in (RUNS/(tag('micro',e['document'])+'.GPU_EXCLUSIVITY.jsonl')).read_text().splitlines()]
        assert len(guards)==len(samples) and all(h['exclusive'] for h in guards)
        overhead=max(0,max(float(h['physical']['stdout'].strip().split(',')[6])*1024**2-m['reserved_bytes'] for h,m in zip(guards,samples)))
        mem=v['gpu_memory']; reserved=mem['peak_reserved_bytes']+1.5*(projection-bn+unexplained*(32768/8448))+overhead
        if reserved>mem['total_bytes']: errors.append(dict(document=e['document'],reason='PROJECTED_ACTUAL_CACHE_WORKSPACE_CAPACITY_UNSAFE',projected=reserved,capacity=mem['total_bytes']))
        worst=max(b['elapsed_seconds']/b['tokens'] for b in s['blocks'] if b['end']>2048)
        rows.append(dict(document=e['document'],gpu=6,gpu_uuid=UUID,status='PASS',micro_canonical_path=str(p),micro_sha256=digest(p),
            endpoint=8448,loading_seconds=v['model_loading_seconds'],total_wall_seconds=v['total_wall_seconds'],blocks=s['blocks'],
            throughput_tokens_per_second=8448/s['elapsed_seconds'],post2048_worst_seconds_per_token=worst,
            FP_step_seconds=s['model_step_fp_seconds'],student_step_seconds=s['model_step_student_seconds'],gpu_memory=mem,
            native_first_bytes=b0,native_8448_bytes=bn,native_bytes_per_token=rate,projected_native_32K_bytes=projection,
            unexplained_non_native_growth_bytes=unexplained,NVML_measured_overhead_bytes=overhead,
            conservative_projected_reserved_bytes=reserved,projected_full_roster_wall_seconds=1.5*(v['total_wall_seconds']+(32768-8448)*worst*(32768/8448)),
            persistent_integrity=v['persistent_integrity'],complete_2048_8192_continuous_QDQ=True,
            prefix512_exact_checked=True,extra8448_repeat_not_run=True,no_scientific_KL_read=True))
    expected=max(64*1024**2,16*sum(Path(r['micro_canonical_path']).stat().st_size for r in rows)); free=shutil.disk_usage(OUT).free
    if free<expected+2*1024**3: errors.append(dict(reason='FULL_OUTPUT_STORAGE_UNSAFE',free_bytes=free,expected_bytes=expected))
    emit(OUT/'MICRO_8448_RESULTS.json',dict(status='PASS',documents=rows,performance_only_no_KL_outcome=True))
    emit(OUT/'PHYSICAL_RESOURCE_FEASIBILITY.json',dict(status='STOP' if errors else 'PASS',documents=rows,errors=errors,
        disk_free_bytes=free,expected_output_bytes=expected,GPU_HOURS_CAP=None,FIXED_VRAM_PERCENT_CAP=None,
        final_mapping='SERIAL_GPU6',four_students_two_FP=True,projection_not_measured32K=True))
    if errors: stop('PHYSICAL_RESOURCE_GATE_STOP','MICRO',errors=errors); raise RuntimeError('PHYSICAL_RESOURCE_GATE_STOP')
    print('V4_MICRO_PHYSICAL_GATE_PASS',flush=True)

def freeze_formal():
    cfg,sel,freeze=verify(); gate=checked(OUT/'PHYSICAL_RESOURCE_FEASIBILITY.json'); assert gate['status']=='PASS'
    qual=checked(OUT/'GPU_512X2_QUALIFICATION.json'); assert qual['status']=='PASS'
    diagnostics=qual['reducer_diagnostics']; assert len(diagnostics)==24
    test=checked(OUT/'V4_PERSISTENT_CPU_CHECK.json'); assert test['status']=='PASS_CPU_ONLY'
    # Same prospective rule as original V3 stages, frozen before qualification:
    # sampled arithmetic resolution, never a universal/statistical bound.
    resolution=10*max([2**-23,test['synthetic_reducer_rounding_difference']]+[s['absolute_arithmetic_difference'] for s in diagnostics])
    emit(OUT/'FROZEN_FORMAL_32K_SPEC.json',dict(status='FROZEN',epoch=time.time(),code_commit=freeze['code_commit'],code_freeze_sha256=digest(FREEZE),
        original_pair_selection_sha256=digest(OLD/'PROSPECTIVE_DOCUMENT_AND_PAIR_SELECTION.json'),
        mapping_amendment_sha256=digest(OUT/'HARDWARE_MAPPING_AMENDMENT.json'),documents=sel['documents'],
        document_launch_order=['pg19-test-860','pg19-test-5956'],execution='SERIAL_GPU6',GPU_uuid=UUID,
        primary=dict(metric='original full-vocabulary FP32 KL(FP||student)',window=[8192,32768],reducer='original Python sum(values)/len(values)'),
        secondary=dict(metric='same trajectory original full-vocabulary FP32 KL(FP||student)',window=[2048,32768],reducer='original Python sum(values)/len(values); auc=sum(values)'),
        indexing='original position0..32767 token[position], same-step live FP/student logits; no shift',input_tokens=32769,endpoint=32768,
        student_count=4,FP_count=2,FP_pairing='same GPU fresh live FP cache plus two independent persistent student caches interleaved per document',
        every_token_QDQ_including_after_anchors=True,carrier='FP32 carrying dequantized INT8-R128 values, not physical resident INT8 or high precision shadow',
        uncertainty_rule=dict(threshold=resolution,method='prefrozen 10*max(qualified same-logit sampled GPU32/CPU64 arithmetic difference, FP32 epsilon, synthetic Python sum rounding)',
            empirical_only_not_universal_error_bound=True,not_statistical_significance=True,samples=diagnostics,
            exact_zero='TIE',abs_delta_le_threshold='UNRESOLVED',negative_beyond_threshold='P1',positive_beyond_threshold='L7',
            resolution_failure='UNRESOLVED; preserve signed Delta; never invent posthoc threshold'),
        no_progress_seconds={r['document']:max(1800,8*max(b['elapsed_seconds'] for b in r['blocks'])*(32768/8448)+300) for r in gate['documents']},
        numeric_prefix_check=dict(boundaries=[512,8448],same_model_step_geometry_chunk_cache=True,full_FP_and_two_student_byte_hashes=True,
            micro_is_single_frozen_execution_no_observer_repeat=True,first_actual_prefix_mismatch='FAIL_CLOSED_NO_RETRY'),
        all_formal_workers_end_before_science_analysis=True,no_first_document_based_plan_change=True,no_retry_or_trajectory_stitching=True,
        qualification_sha256=digest(OUT/'GPU_512X2_QUALIFICATION.json'),physical_gate_sha256=digest(OUT/'PHYSICAL_RESOURCE_FEASIBILITY.json'),
        GPU_HOURS_CAP=None,FIXED_VRAM_PERCENT_CAP=None))
    print('V4_FORMAL_SPEC_FROZEN',resolution,flush=True)
