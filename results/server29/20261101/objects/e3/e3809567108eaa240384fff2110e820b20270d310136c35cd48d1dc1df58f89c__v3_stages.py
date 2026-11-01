"""Mechanical micro/physical capacity gate and prospective formal prefreeze."""
import math
import shutil
import time
from pathlib import Path
from v3_common import *

def physical_micro_gate(parallel):
    cfg,sel,freeze=verify(); rows=[]; errors=[]
    for e in sel['documents']:
        v,p=complete('micro',e['document'],e['gpu']); s=v['stream']; assert 'results' not in s,'MICRO_KL_BLINDNESS_FAIL'
        assert s['endpoint']==8448 and s['every_token_qdq_including_after_anchors']
        inv=s['hybrid_cache_inventories']; assert all(str(x) in inv for x in (2047,2048,8191,8192,8447))
        # MLA cache growth follows the exact frozen inventory's sequence dimension;
        # constant KDA/conv components are not treated as attention growth.
        first,last=inv['0'],inv['8447']; total_first=sum(x['cache_bytes'] for x in first.values()); total_last=sum(x['cache_bytes'] for x in last.values())
        native_per_token=(total_last-total_first)/8447
        assert native_per_token>=0
        projection_native=total_last+(32768-8448)*native_per_token
        samples=v['progress_memory_storage']; assert samples and samples[-1]['position']==8448
        # At every inventory token, MLA is linear and recurrent/conv layouts fixed.
        for at,branches in inv.items():
            observed=sum(x['cache_bytes'] for x in branches.values())
            assert observed==total_first+int(at)*native_per_token,'UNEXPLAINED_NATIVE_CACHE_GROWTH'
        endpoints=[int(x) for x in inv]
        allocated=v['gpu_memory']['peak_allocated_bytes']; reserved=v['gpu_memory']['peak_reserved_bytes']; capacity=v['gpu_memory']['total_bytes']
        extra_native=projection_native-total_last
        # Project actual native/workspace growth and measured driver overhead;
        # no invented absolute allowance or fixed percentage threshold.
        growth=[x['allocated_bytes'] for x in samples if x['position']>=2048]
        native_span=(8448-2048)*native_per_token
        unexplained=max(0,growth[-1]-growth[0]-native_span) if growth else 0
        guard_rows=[json.loads(x) for x in (RUNS/(tag('micro',e['document'],e['gpu'],0)+'.GPU_EXCLUSIVITY.jsonl')).read_text().splitlines()]
        assert len(guard_rows)==len(samples) and all(x['exclusive'] for x in guard_rows)
        overhead=max(0,max(float(h['physical']['stdout'].strip().split(',')[6])*1024**2-sample['reserved_bytes'] for h,sample in zip(guard_rows,samples)))
        # Non-native growth is conservatively extrapolated, not rejected merely
        # for exceeding an invented absolute/percentage memory threshold.
        projected_reserved=reserved+1.5*(extra_native+unexplained*(32768/8448))+overhead
        if projected_reserved>capacity: errors.append(dict(document=e['document'],reason='CONSERVATIVE_NATIVE_AND_WORKSPACE_CAPACITY_NOT_SAFE',projected_bytes=projected_reserved,physical_bytes=capacity))
        post=[x for x in s['blocks'] if x['end']>2048]; worst=max(x['elapsed_seconds']/x['tokens'] for x in post)
        projected_wall=1.5*(v['total_wall_seconds']+(32768-8448)*worst*(32768/8448))
        rows.append(dict(document=e['document'],gpu=e['gpu'],gpu_uuid=v['gpu_uuid'],micro_canonical_path=str(p),micro_sha256=digest(p),
            endpoint=8448,status='PASS',loading_seconds=v['model_loading_seconds'],total_wall_seconds=v['total_wall_seconds'],
            blocks=s['blocks'],post2048_worst_seconds_per_token=worst,throughput_aggregate_tokens_per_second=8448/s['elapsed_seconds'],
            FP_runtime_seconds=s['model_step_fp_seconds'],student_runtime_seconds=s['model_step_student_seconds'],
            peak_allocated_bytes=allocated,peak_reserved_bytes=reserved,capacity_bytes=capacity,native_cache_first_bytes=total_first,
            native_cache_8448_bytes=total_last,native_cache_linear_bytes_per_token=native_per_token,
            projected_native_cache_32k_bytes=projection_native,conservative_projected_reserved_bytes=projected_reserved,
            observed_unexplained_non_native_growth_bytes=unexplained,conservative_projected_full_roster_wall_seconds=projected_wall,
            NVML_measured_driver_context_overhead_bytes=overhead,
            persistent_integrity=v['persistent_integrity'],performance_only_no_KL_outcome_access=True))
    free=shutil.disk_usage(OUT).free
    expected_storage=max(64*1024**2,16*sum(Path(r['micro_canonical_path']).stat().st_size for r in rows))
    if free<expected_storage+2*1024**3: errors.append(dict(reason='COMPLETE_OUTPUT_STORAGE_NOT_SAFE',free=free,needed=expected_storage+2*1024**3))
    emit(OUT/'MICRO_8448_RESULTS.json',dict(status='PASS',documents=rows,full_frozen_FP_plus_two_student_roster=True,
        continued_QDQ_across_2048_8192=True,prospective_KL_rank_not_published_or_used=True))
    emit(OUT/'PHYSICAL_RESOURCE_FEASIBILITY.json',dict(status='PASS' if not errors else 'STOP',documents=rows,
        errors=errors,disk_free_bytes=free,projected_complete_output_bytes=expected_storage,
        GPU_HOURS_CAP=None,FIXED_VRAM_PERCENT_CAP=None,parallel_documents=parallel,entire_four_students_and_two_FP=True,
        projection_is_conservative_not_measured32K=True))
    if errors:
        stop('PHYSICAL_RESOURCE_FEASIBILITY_STOP','micro_resource',errors=errors)
        raise RuntimeError('PHYSICAL_RESOURCE_FEASIBILITY_STOP')
    return rows

def freeze_formal(parallel):
    cfg,sel,freeze=verify(); gate=checked(OUT/'PHYSICAL_RESOURCE_FEASIBILITY.json'); assert gate['status']=='PASS'
    q=checked(OUT/'GPU_512X2_QUALIFICATION.json'); assert q['status']=='PASS'
    samples=q['reducer_diagnostics']; assert len(samples)==12
    assert all(math.isfinite(x['absolute_arithmetic_difference']) for x in samples)
    cpu=checked(OUT/'CPU_REGRESSION_RESULTS.json')
    resolution=10*max([2**-23,cpu['synthetic_reducer_rounding_difference']]+[x['absolute_arithmetic_difference'] for x in samples])
    emit(OUT/'FROZEN_FORMAL_32K_SPEC.json',dict(status='FROZEN',epoch=time.time(),code_commit=freeze['code_commit'],
        code_freeze_sha256=digest(FREEZE),source_identity_sha256=digest(OUT/'SOURCE_AND_FROZEN_SELECTION_IDENTITY.json'),
        original_pair_selection_sha256=digest(OLD/'PROSPECTIVE_DOCUMENT_AND_PAIR_SELECTION.json'),documents=sel['documents'],
        primary=dict(metric='full-vocabulary FP32 KL(FP||student)',window=[8192,32768],reducer='Python sum(values)/len(values), frozen summarize'),
        secondary=dict(metric='same trajectory full-vocabulary FP32 KL(FP||student)',window=[2048,32768],reducer='same summarize mean; auc=sum(values)'),
        token_indexing='model_step token[position] at position0..32767; matched FP/student logits of that same step; no logit shift introduced',
        input_tokens=32769,endpoint=32768,conditions='original pair condition_a and condition_b, sorted exactly as immutable engine',
        FP_pairing='one fresh live same-GPU full-length FP cache interleaved with two independent persistent student caches per document',
        student_count=4,FP_count=2,parallel_documents=parallel,document_launch_order=[6,3],
        every_token_INT8_R128_QDQ_inverse_native_writeback=True,carrier='FP32 containing dequantized INT8 values, not unquantized shadow or physical INT8 storage',
        uncertainty_rule=dict(threshold=resolution,method='10*max(same-qualified-logit sampled GPU32-vs-CPU64 KL arithmetic difference,FP32 epsilon,CPU Python-reducer rounding)',
            sample_positions=[0,255,511],samples=samples,empirical_not_universal_error_bound=True,not_statistical_significance=True,
            exact_zero='TIE',abs_delta_le_threshold='NUMERICALLY_UNRESOLVED',negative_beyond_threshold='P1',positive_beyond_threshold='L7'),
        no_progress_seconds={r['document']:max(1800,8*max(b['elapsed_seconds'] for b in r['blocks'])*(32768/8448)+300) for r in gate['documents']},
        automatic_retry=False,exact_resume_qualified=False,first_failure_action='STOP_AND_REPORT_INCOMPLETE_NO_TRAJECTORY_STITCHING',
        analysis_after_both_documents_end=True,GPU_HOURS_CAP=None,FIXED_VRAM_PERCENT_CAP=None,
        physical_gate_sha256=digest(OUT/'PHYSICAL_RESOURCE_FEASIBILITY.json'),qualification_sha256=digest(OUT/'GPU_512X2_QUALIFICATION.json'),
        model_rotation_data_quantizer_and_numerical_tolerance_unchanged=True))
    print('V3_FORMAL_PREFREEZE_READY',resolution,flush=True)
