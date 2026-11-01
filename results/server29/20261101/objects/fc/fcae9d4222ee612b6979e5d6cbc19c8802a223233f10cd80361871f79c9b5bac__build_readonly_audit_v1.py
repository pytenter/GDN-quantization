"""Generate the six requested audit deliverables from captured server evidence.

CPU arithmetic/serialization only. No imports from any experiment or ML runtime.
Never writes into, executes, or repairs the original experiment.
"""
import ast
import datetime as dt
import hashlib
import itertools
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent
E = json.loads((OUT/'READONLY_SERVER_EVIDENCE.json').read_text(encoding='utf-8'))
S = json.loads((OUT/'READONLY_SERVER_SUPPLEMENT.json').read_text(encoding='utf-8'))
ROOT = E['original_root']
D = E['data']
CFG = D['PHASE_B_CALIBRATION_CONFIG.json']
SEL = D['PROSPECTIVE_DOCUMENT_AND_PAIR_SELECTION.json']
METRICS = ['P1','P2_W128','P3_H128','B0','L7_FUNCTIONAL_ORIGINAL','L7_STATE_ORIGINAL','L7_TOTAL_ORIGINAL']
FLAGS = dict(NO_NEW_GPU_SCIENTIFIC_RUN=True, NO_NEW_32K_LABELS=True, NO_TRAINING=True)
VERDICT = 'ELIGIBLE_FOR_ISOLATED_REPAIR'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def content_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def stamp(epoch):
    return dt.datetime.fromtimestamp(epoch,dt.timezone(dt.timedelta(hours=8))).isoformat(timespec='microseconds')

def identity(name):
    return E['identities'][name if name.startswith('/') else ROOT+'/'+name]

def write(name,value):
    with (OUT/name).open('x',encoding='utf-8',newline='\n') as f:
        if isinstance(value,str): f.write(value.rstrip()+'\n')
        else: json.dump(value,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False); f.write('\n')

def ranks(values):
    # Independent exact-tie midranks, not importing/re-running the old selector.
    return {c:1+sum(x<v for x in values.values())+(sum(x==v for x in values.values())-1)/2
            for c,v in values.items()}

def frozen_order_audit(panel):
    a,b = panel['P1'],panel['L7_TOTAL_ORIGINAL']; ra,rb=ranks(a),ranks(b)
    rows=[]
    for ca,cb in itertools.combinations(sorted(a),2):
        ga,gb=abs(ra[ca]-ra[cb]),abs(rb[ca]-rb[cb])
        if (a[ca]-a[cb])*(b[ca]-b[cb]) < 0 and ga>=3 and gb>=3:
            rows.append(dict(condition_a=ca,condition_b=cb,rank_gap_a=ga,rank_gap_b=gb,
                minimum_rank_gap=min(ga,gb),metric_a_winner=ca if a[ca]<a[cb] else cb,
                metric_b_winner=ca if b[ca]<b[cb] else cb))
    return sorted(rows,key=lambda x:(-x['minimum_rank_gap'],x['condition_a'],x['condition_b']))

def restore_fp(x):
    return dict(drivers=x['drivers'],checkpoint_states={int(k):v for k,v in x['checkpoint_states'].items()},
        full_logits={int(k):v for k,v in x['full_logits'].items()},
        post_o_proj={int(k):{int(l):v for l,v in d.items()} for k,d in x['post_o_proj'].items()})

def main():
    assert sha(OUT/'READONLY_SERVER_EVIDENCE.json')=='24e5af27dd1ec2b482779e53bcb0f91862b132d3bd293f71844ecff88f1105d2'
    assert sha(OUT/'READONLY_SERVER_SUPPLEMENT.json')=='ac28344edc9ff49f905a7e85488c648eb9a83ec202e395fe1e42fd69d60297c2'
    assert S['base_evidence_sha256']==sha(OUT/'READONLY_SERVER_EVIDENCE.json')
    assert E['actual_server_evidence'] and S['actual_server_evidence']
    assert all(x['passed'] for x in E['hash_checks'])
    assert not E['original_files_changed_or_added_or_removed'] and S['original_files_still_identical']
    assert all(x['metadata_matches_original'] for x in E['model_weights'])
    assert all(x['equals_run_file'] for x in E['git']['persistent_tree_file_comparison'].values())
    assert all(x['existence']['returncode']==0 for x in E['git']['commits'].values())
    assert E['resources']['torch_cuda_fla_transformers_not_imported'] and S['no_gpu_imports']
    tables=D['PHASE_B_CALIBRATION_MATCHED_TABLES.json']; scores=tables['scores']; original_ranks=tables['ranks']
    assert len(scores)==96 and len(original_ranks)==672
    indexed={(x['document'],x['window'],x['condition']):x for x in scores}
    assert len(indexed)==96
    rank_index={(x['document'],x['window'],x['condition'],x['metric']):x for x in original_ranks}
    assert len(rank_index)==672
    fresh_inputs={x['document_id']:x for x in D['PROSPECTIVE_DOCUMENT_SELECTION.json']['selected_documents']}
    prior_ids=set(D['PRIOR_DOCUMENT_PATH_IDENTITY_AUDIT.json']['prior_measurement_path_ids'])
    conditions=CFG['conditions']; all_panels={}; row_checks=0; rank_checks=0; canonical_checks=[]
    for entry in SEL['documents']:
        doc=entry['document']; gpu=entry['gpu']; run=E['calibration'][f'formal_{doc}_gpu{gpu}.json']
        assert doc not in prior_ids and doc not in {'pg19-test-37702','pg19-test-5396','pg19-test-48693','pg19-test-51410'}
        assert run['valid'] and len(run['results'])==16 and [x['condition'] for x in run['results']]==conditions
        assert run['runner_sha256']==identity('run_phase_b_calibration_v2.py')['sha256']
        assert run['config_sha256']==identity('PHASE_B_CALIBRATION_CONFIG.json')['sha256']
        assert run['freeze_sha256']==identity('PHASE_B_CALIBRATION_CODE_FREEZE.json')['sha256']
        assert run['gpu_uuid']==CFG['gpu_uuids'][str(gpu)]
        assert not run['long_horizon_outcome_generated_or_used'] and not run['prior_score_set_joined']
        assert not run['gradients_or_optimizer_used']
        fp=run['fp_reference']; actual_fp=content_sha(restore_fp(fp['tensor_identities']))
        assert actual_fp==fp['content_sha256']==entry['calibration_identity']['fp_reference_content_sha256']
        assert fp['tokens']==1152 and fp['replay_parity']['passed']
        assert len(fp['helper_real_forward_checks'])==72
        assert all(x['bitwise_equal'] and x['identity_loss']==0 for x in fp['helper_real_forward_checks'])
        for result in run['results']:
            gates=result['mechanical_gates']
            assert gates['passed'] and gates['every_token_layer_qdq_count']==20736
            assert gates['actual_next_kernel_equal_post_qdq_count']==gates['expected_consumption_count']==20718
            for window in ['256','1024','MEAN_256_1024']:
                row=indexed[(doc,window,result['condition'])]
                ref=result['aggregate'] if window=='MEAN_256_1024' else next(x for x in result['windows'] if x['start']==int(window))
                assert row['fp_reference_sha256']==actual_fp and row['gpu_uuid']==run['gpu_uuid']
                assert row['input_sha256']==fresh_inputs[doc]['token_file']['sha256']
                for m in METRICS:
                    assert row[m]==ref[m] and math.isfinite(row[m]); row_checks+=1
                    mean=sum(x[m] for x in result['windows'])/2
                    assert math.isclose(result['aggregate'][m],mean,rel_tol=1e-14)
                assert math.isclose(row['L7_TOTAL_ORIGINAL'],row['L7_FUNCTIONAL_ORIGINAL']+.1*row['L7_STATE_ORIGINAL'],rel_tol=2e-6)
        for window in ['256','1024','MEAN_256_1024']:
            panel={m:{c:indexed[(doc,window,c)][m] for c in conditions} for m in METRICS}
            all_panels[(doc,window)]=panel
            for m,vals in panel.items():
                recomputed=ranks(vals)
                for c in conditions:
                    old=rank_index[(doc,window,c,m)]
                    assert old['value']==vals[c] and old['midrank']==recomputed[c]; rank_checks+=1
        candidates=frozen_order_audit(all_panels[(doc,'MEAN_256_1024')])
        assert candidates==entry['eligible_pairs_in_frozen_order'] and candidates[0]==entry['pair']
        for m in METRICS:
            values=all_panels[(doc,'MEAN_256_1024')][m]
            ca,cb=entry['pair']['condition_a'],entry['pair']['condition_b']
            assert entry['pair_scores'][m]=={ca:values[ca],cb:values[cb]}
            assert entry['predicted_winners'][m]==(ca if values[ca]<values[cb] else cb)
        raw=fresh_inputs[doc]
        order=hashlib.sha256((CFG['experiment']+'|seed=0|'+doc+'|'+raw['token_file']['sha256']).encode()).hexdigest()
        assert order==raw['order_sha256'] and E['tokens'][doc]['count']==32769 and E['tokens'][doc]['legal_vocab']
        canonical_checks.append(dict(document=doc,fp_identity_recomputed=actual_fp,all16_roster_complete=True,
            helper_checks=72,qdq_writes_each_rotation=20736,next_kernel_consumptions_each_rotation=20718,
            score_cells_recovered_from_existing_canonical=16*3*7,selection_order_hash_recomputed=order))
    assert SEL['route']=='A' and SEL['comparator']==['P1','L7_TOTAL_ORIGINAL']
    assert not SEL['new_long_outcomes_generated_or_read'] and not SEL['long_run_authorized']
    assert D['PHASE_B_PAIR_METHODS_FREEZE.json']['epoch'] < min(
        E['calibration'][f'formal_{x["document"]}_gpu{x["gpu"]}.LAUNCH_AUTHORIZATION.json']['epoch'] for x in SEL['documents'])
    assert max(E['calibration'][f'formal_{x["document"]}_gpu{x["gpu"]}.json']['completed_epoch'] for x in SEL['documents'])<SEL['epoch']
    assert SEL['epoch']<min(v['epoch'] for k,v in E['persistent'].items() if k.endswith('.LAUNCH_AUTHORIZATION.json'))
    assert not S['new_persistent_canonical_paths'] and not S['complete_cost_gate_exists']
    provenance=dict(server='zypan@202.38.247.29:22',original_root=ROOT,remote_audit_root=E['output_root'],
        local_audit_root=str(OUT),actual_server_collection_started=stamp(E['started_epoch']),
        actual_server_collection_completed=stamp(E['completed_epoch']),supplement_completed=stamp(S['epoch']),
        evidence_sha256=sha(OUT/'READONLY_SERVER_EVIDENCE.json'),supplement_sha256=sha(OUT/'READONLY_SERVER_SUPPLEMENT.json'),
        builder_sha256=sha(__file__),git=E['git'],source_repositories=S['source_repositories'],
        all_hash_checks=E['hash_checks'],identities=E['identities'],supplement_hashes=S['hashes'],
        original_root_files_unchanged=len(E['original_inventory_before']),
        historical_panel_coverage='80 canonical + 80 sidecars + 2 permanent STOP records fully rehashed; no claim of a separately verified 128 panel',
        weights_coverage='32 full weight shard SHA256 rehashes, not just metadata checks',
        cpu_only_no_ml_runtime_imports=True,gpu_hours_this_audit=0,**FLAGS)
    full_order={}
    for (doc,window),panel in all_panels.items():
        full_order.setdefault(doc,{})[window]={m:[dict(condition=c,midrank=ranks(vals)[c],score=vals[c])
            for c in sorted(vals,key=lambda c:(vals[c],c))] for m,vals in panel.items()}
    selection_audit=dict(task='LING_KDA_P1_L7_PHASEB_STOP_FORENSIC_AUDIT_V1',status='FROZEN_SELECTION_VALID_NO_RESELECTION',
        cohort_distinction=dict(A_phase_a=['pg19-test-37702','pg19-test-5396'],
            B_prior_p1_b0=['pg19-test-48693','pg19-test-51410'],C_this_frozen_phase_b=list(CFG['documents'])),
        frozen_selection=SEL,prospective_inputs=D['PROSPECTIVE_DOCUMENT_SELECTION.json']['selected_documents'],
        dataset_identity=dict(pool=D['PHASE_B_DOCUMENT_SELECTION_RULE_FREEZE.json']['pool'],split='PG19 test document IDs',
            version_basis='Local phase3_mixed_hardware_v1_20261003 snapshot paths plus exact raw/token content SHA256',
            upstream_dataset_revision='NOT_PRESENT_IN_AUDITED_FREEZE; not inferred from folder date',
            fresh_token_checks=E['tokens'],reencoding='Historical frozen exact reencoding evidence rehashed; not re-tokenized during this audit'),
        all16_rotation_roster=D['SOURCE_IDENTITIES.json']['rotations'],
        model_identity=D['SOURCE_IDENTITIES.json']['model_identity'],
        calibration=dict(window_starts=[256,1024],horizon=128,scored_ranges=[[256,384],[1024,1152]],
            aggregation='Equal arithmetic mean of the two window scores; all18 KDA layers and all16 heads',
            original_L7='L7_FUNCTIONAL_ORIGINAL + 0.1 * L7_STATE_ORIGINAL; post-o_proj original helpers',
            recurrence_note='Original L7 already H128 recurrent-aware; matched deployment application is not replay of historical 512-token training sampling',
            P1_definition='Unchanged inherited query_readout(prepared query, native-coordinate state error).square().mean; no new normalization invented',
            P1_source=identity('/data/zypan/experiment_runs/LING_KDA_SHORT_HORIZON_SURROGATE_VALIDATION_V1__resume20261010_v2/run_short_horizon_proxy_v2.py'),
            matched_fresh_same_gpu_FP_identity_checks=canonical_checks),
        existing_all_scores=scores,existing_all_ranks=original_ranks,complete_sorted_rankings=full_order,
        independent_audit=dict(score_cells_checked=row_checks,midrank_cells_checked=rank_checks,
            exact_ties_average_midrank=True,all_eligible_pairs_in_frozen_order_match=True,
            both_selected_pair_gaps_at_least_3=True,opposite_predictions=True,
            selection_rule_audited_without_reexecuting_selector=True),
        leakage_audit=dict(status='NO_OUTCOME_LEAKAGE_FOUND_IN_AUDITED_ASSETS',
            methods_freeze_before_formal_calibration=True,selection_after_both_complete_calibrations=True,
            selection_before_first_persistent_launch=True,selection_epoch=SEL['epoch'],
            persistent_engine_before_forward_failed=True,no_long_canonical_or_micro_or_full_cost_gate=True,
            selector_static_inputs='Only complete short calibration, frozen rules/route, source identities, accounting and input metadata',
            caveat='Cannot prove all undocumented human/file-access history; claims limited to frozen code, receipts, inventories and audit evidence'),
        next_long_run_authorized=False,provenance=provenance,**FLAGS)

    events=[]; failures=[]
    worker=E['static_sources'][ROOT+'/run_persistent_stage_v1.py']
    assert worker.splitlines()[62].strip()=='assert torch.are_deterministic_algorithms_enabled()'
    assert not any('use_deterministic_algorithms' in l or 'set_deterministic_debug_mode' in l for l in worker.splitlines()[:63])
    for entry in SEL['documents']:
        tag=f'qualification_{entry["document"]}_gpu{entry["gpu"]}'
        rec=E['persistent'][tag+'.LAUNCH_AUTHORIZATION.json']; st=E['persistent'][tag+'.STARTED.json']; stop=E['persistent'][tag+'.STOP.json']
        assert stop['error']=='AssertionError()' and 'line 63' in stop['traceback'] and 'are_deterministic_algorithms_enabled' in stop['traceback']
        assert stop['runner_sha256']==identity('run_persistent_stage_v1.py')['sha256']
        log=E['persistent'][tag+'.log']
        assert '32/32' in log and 'persistent_progress' not in log and '"event": "complete"' not in log
        bound=dict(started_epoch=st['started_epoch'],stopped_epoch=stop['stopped_epoch'],
            started_time=stamp(st['started_epoch']),stopped_time=stamp(stop['stopped_epoch']))
        events.extend([dict(epoch=rec['epoch'],timestamp=stamp(rec['epoch']),gate='prelaunch identity/device/reservation',
            document=entry['document'],status='PASS_AS_RECORDED',input='freeze and selection SHA, UUID, idle device, complete 900s roster reservation',
            output='LAUNCH_AUTHORIZATION',evidence=identity('runs_b_persistent/'+tag+'.LAUNCH_AUTHORIZATION.json')),
            dict(epoch=st['started_epoch'],timestamp=stamp(st['started_epoch']),gate='worker verify + argument/receipt/deadline',
            document=entry['document'],status='PASS_BY_REACHABILITY',input='frozen selection/config/receipt',output='STARTED',
            evidence=identity('runs_b_persistent/'+tag+'.STARTED.json'))])
        failures.append(dict(document=entry['document'],gpu=entry['gpu'],gpu_uuid=rec['gpu_uuid'],pid=st['pid'],
            **bound,gate='GLOBAL_TORCH_DETERMINISTIC_ALGORITHMS_ENABLED_ASSERTION',source_line=63,
            source=identity('run_persistent_stage_v1.py'),original_stop=identity('runs_b_persistent/'+tag+'.STOP.json'),
            original_log=identity('runs_b_persistent/'+tag+'.log'),original_traceback=stop['traceback'],
            expected=True,actual=False,observation_basis='False inferred from original failed boolean assert and installed getter source; not a separately logged flag value',
            stream_tokens_processed=0,status='FAIL_AND_STOP_NO_RETRY'))
        events.append(dict(epoch=stop['stopped_epoch'],timestamp=stamp(stop['stopped_epoch']),gate=failures[-1]['gate'],
            document=entry['document'],status='FAIL',input='Global PyTorch deterministic flag',output='AssertionError; exclusive STOP',
            evidence=failures[-1]['original_stop']))
    failures.sort(key=lambda x:x['stopped_epoch']); events.sort(key=lambda x:x['epoch'])
    failure_report=dict(task=selection_audit['task'],root_cause_status='ESTABLISHED_FROM_ORIGINAL_TRACEBACK_AND_HASH_BOUND_SOURCE',
        first_failure=failures[0],both_failures=failures,original_event_timeline=events,
        gates_between_STARTED_and_first_failure=[
            dict(gate='GPU UUID/device exclusivity and kernel/driver/Torch/CUDA/selector',status='PASS_BY_CONTROL_FLOW_REACHABILITY',source_lines=[49,50]),
            dict(gate='peer STOP/deadline guard',status='PASS_BY_CONTROL_FLOW_REACHABILITY',source_lines=[51,54]),
            dict(gate='load_assets + token/raw/rotation identity and CPU shape checks',status='RETURNED_WITHOUT_EXCEPTION',source_lines=[55,58]),
            dict(gate='seed setting + Capture model construction and 32/32 shard load',status='RETURNED_WITHOUT_EXCEPTION',source_lines=[59,62]),
            dict(gate='global deterministic algorithms flag',status='FAIL',source_lines=[63])],
        timestamp_precision_note='Intermediate per-gate timestamps not logged. These gates are ordered by frozen control flow and bounded by STARTED/STOP; no invented exact timestamps or GPU PASS.',
        not_reached=['TF32/CUBLAS assertions following line63','rotation move and peak-memory reset','stream() first forward',
            '512-token FP replay/full-logit parity','persistent every-token QDQ/inverse/writeback/next-kernel consumption',
            'hybrid KDA/MLA/conv cache numerical qualification','finite/repeat equality','8448 microbenchmark','complete cost gate','formal32768'],
        classification='ENGINEERING_RUNTIME_CONFIGURATION_PRECONDITION_MISMATCH',
        excluded_conclusions=['Not a measured numerical nondeterminism failure','Not an established QDQ/scientific metric failure',
            'Not an observed OOM/cache capacity failure','Not budget exhaustion','No new long-horizon winner label'],
        static_explanation='Worker asserts the global Torch flag after Capture but contains no explicit enabling call before the assertion. Fixed FLA selector, seed and CUBLAS environment do not themselves assert or guarantee this distinct global flag. The original trace establishes the flag was not enabled there; complete dynamic provenance of every legacy seed helper is not needed or claimed.',
        installed_torch_getter_definition=next(v for p,v in E['static_sources'].items() if p.endswith('/torch/__init__.py')),
        shared_latch=D['PERSISTENT_FAIL_CLOSED_STOP.json'],shared_latch_identity=identity('PERSISTENT_FAIL_CLOSED_STOP.json'),
        later_guard_exception=S['files'][ROOT+'/PERSISTENT_GUARD_EXCEPTION_1791702534453816856.json'],
        later_guard_is_not_first_failure=True,partial_latch_artifacts_retained=[p for p in E['original_inventory_before'] if '.partial.' in p],
        earlier_CPU_build_errors='Separate pre-GPU engineering appendices preserved. Not the first failure of the persistent qualification execution.',
        repair_candidate='Explicit deterministic runtime initialization and flag telemetry, preserving the assertion; isolated new code/root only if separately authorized',
        repair_executed=False,full_new_qualification_required_before_any_long_run=True,provenance=provenance,**FLAGS)

    ledger=D['final_stop_20261011/GPU_COST_LEDGER.json']; prior=ledger['prior_a_and_b_calibration']; cost_rows=[]
    for phase,rows in [('A',prior['phase_a']['gpu_runs']),('B_calibration',prior['phase_b_calibration_runs'])]:
        for row in rows:
            tag=row['tag']
            if phase=='A':
                rec=S['files'][ROOT+'/runs/'+tag+'.LAUNCH_AUTHORIZATION.json']; canon=S['files'][ROOT+'/runs/'+tag+'.json']
            else:
                rec=E['calibration'][tag+'.LAUNCH_AUTHORIZATION.json']; canon=E['calibration'][tag+'.json']
            value=(canon['completed_epoch']-rec['epoch'])/3600
            assert math.isclose(value,row['gpu_hours_upper_bound'],rel_tol=1e-14,abs_tol=1e-15)
            cost_rows.append(dict(stage=phase,tag=tag,launch_epoch=rec['epoch'],completed_epoch=canon['completed_epoch'],
                gpu_uuid=row['gpu_uuid'],device_hours_including_startup=value))
    for row in ledger['persistent_runs']:
        tag=row['tag']; rec=E['persistent'][tag+'.LAUNCH_AUTHORIZATION.json']; stop=E['persistent'][tag+'.STOP.json']
        value=(stop['stopped_epoch']-rec['epoch'])/3600
        assert math.isclose(value,row['gpu_hours_upper_bound'],rel_tol=1e-14)
        cost_rows.append(dict(stage='B_persistent_qualification',tag=tag,launch_epoch=rec['epoch'],completed_epoch=stop['stopped_epoch'],
            gpu_uuid=row['gpu_uuid'],device_hours_including_startup=value))
    total=sum(x['device_hours_including_startup'] for x in cost_rows)
    release=prior['total_gpu_hours_upper_bound']+sum((ledger['first_verified_process_release_observation_epoch']-x['launch_epoch'])/3600 for x in ledger['persistent_runs'])
    assert math.isclose(total,ledger['total_gpu_hours_upper_bound'],rel_tol=1e-14)
    assert math.isclose(release,ledger['conservative_device_release_gpu_hours_upper_bound'],rel_tol=1e-14)
    hw=E['resources']['hardware']; idle={}
    for gpu,data in hw.items():
        q=data['gpu_query']; apps=data['compute_apps']; holders=data['device_holders']; columns=[x.strip() for x in q['stdout'].strip().split(',')]
        assert q['returncode']==0 and columns[1]==CFG['gpu_uuids'][gpu] and columns[3]==CFG['runtime']['driver']
        idle[gpu]=dict(uuid=columns[1],model=columns[2],driver=columns[3],memory_total_mib=int(columns[4]),
            memory_used_mib=int(columns[5]),gpu_utilization_percent=int(columns[6]),
            no_compute_apps=apps['returncode']==0 and not apps['stdout'].strip(),
            no_device_holders=holders['returncode']==1 and not holders['stdout'].strip() and not holders['stderr'].strip())
    assert E['resources']['kernel']==CFG['runtime']['kernel'] and E['resources']['packages']['torch']==CFG['runtime']['torch']
    assert E['resources']['packages']['nvidia-cuda-runtime-cu12']=='12.8.57'
    resource_report=dict(task=selection_audit['task'],resource_snapshot=E['resources'],resource_observed_time=stamp(E['resources']['observed_epoch']),
        target_GPU_snapshot=idle,persistent_PIDs=S['persistent_processes'],persistent_tmux=S['persistent_tmux'],
        current_runtime_identity_matches_frozen_asserted_fields=True,
        cuda_identity_limit='Torch +cu128 and CUDA runtime distribution12.8.57 obtained without importing Torch; frozen GPU host receipts/control-flow checked runtime12.8. No current forward/runtime qualification was executed.',
        FLA_distribution='fla-core0.5.2; flash-linear-attention distribution name not installed, not a claim that fla module is missing',
        current_git=E['git'],source_repository_git=S['source_repositories'],
        source_identity_scope='Frozen code/dependency file SHA256 match, and14 measurement files match f1fa commit; this is not a claim every unbound runtime file is qualified',
        disk_assessment=dict(result_directory_free_bytes=E['resources']['disks'][ROOT]['free_bytes'],
            original_result_free_threshold_bytes=2*1024**3,original_threshold_currently_satisfied=E['resources']['disks'][ROOT]['free_bytes']>=2*1024**3,
            scratch_paths_missing=['/scratch','/data/zypan/scratch'],
            available_observed_tmp_bytes=E['resources']['disks']['/tmp']['free_bytes'],
            available_triton_cache_bytes=E['resources']['disks']['/data/zypan/.triton/cache']['free_bytes'],
            caveat='Space is a snapshot, not a reservation. No cleanup performed; exact future temporary/cache demand and continuity still require qualification.'),
        accounting=dict(original_ledger=identity('final_stop_20261011/GPU_COST_LEDGER.json'),
            independent_receipt_to_completion_or_stop_recalculation=cost_rows,
            actual_device_time_gpu_hours=total,conservative_process_release_bound_gpu_hours=release,
            remaining_against_actual_gpu_hours=16-total,remaining_against_release_bound_gpu_hours=16-release,
            cap_gpu_hours=16,audit_gpu_hours=0,initial_zero_ledger_not_used_as_current_total=True,
            waiting_other_experiments_CPU_excluded=True,release_bound_is_uncertainty_not_extra_measured_work=True),
        historical_cap_not_exhausted=True,arithmetic_headroom_not_continuation_authority=True,
        four_32k_plus_two_FP_projection=dict(status='NOT_YET_ESTIMABLE_FROM_REQUIRED_QUALIFIED_MICROBENCHMARK',
            actual_micro_count=0,formal_32k_units=0,new_32k_FP_references=0,complete_cost_gate_present=False,
            exact_frozen_cost_rule='Per document: ceil(1.5*(micro.total_wall_seconds + (32768-8448)*max(post2048 block seconds/token)*32768/8448)+30)/3600; includes liveFP + two student streams',
            cache_rule='peak_allocated + native cache bytes*(32768/8448-1) <= 0.85*physical VRAM',
            gate='All previous costs + both new full512 repeat qualifications + both8448 micros + both complete32768 projections <=16; no partial roster',
            previous_other_runner_speed_used=False,qualification_wall_reservation_only='Two900s caps=0.5 GPUh, not a forecast or proof of future qualification'),
        continuation_feasibility='Potential isolated setup repair has arithmetic/resource room, but full scientific continuation remains UNQUALIFIED and no validated complete cost forecast exists',
        provenance=provenance,**FLAGS)

    write('PHASEB_FROZEN_SELECTION_AUDIT.json',selection_audit)
    write('FIRST_FAILURE_GATE_REPORT.json',failure_report)
    write('RESOURCE_AND_BUDGET_AUDIT.json',resource_report)
    next_conditions=[
        '另行获得工程修复与资格执行授权；新隔离目录/新提交/新 freeze，不删除或清除旧 STOP/latch，不复用旧启动标签。',
        '只改 runtime 初始化与可观察性：显式设置并记录确定性配置，保留原 assert，不使用 warn_only 绕过，不更换 Torch/CUDA/FLA 或固定 kernel selector。',
        '验证 frozen document/token/rotation/model/selection SHA、全部配置及同一固定 GPU UUID；空闲与独占要在新启动前再次检查。',
        '两卡分别完整重做新的512-token×2 GPU qualification：同 GPU 新 FP reference/replay与 full-logit parity、真实INT8 codes/inverse/native-coordinate writeback、next-kernel consumption、storage separation、全18KDA/6MLA/conv缓存、finite/repeat。CPU PASS 不替代这些门禁。',
        '两篇都通过后才允许同引擎8448 micro；必须跨2048/8192边界继续每token QDQ，量测loading/JIT/FP+两student总耗时、峰值VRAM、native cache growth、storage与连续性。',
        '累计全部历史+新资格+micro开销，用原1.5安全系数与增长包络计算完整四student32K+两同GPU FP成本；总保守值<=16、VRAM<=85%及存储/连续性都PASS才可申请/执行正式长程。不得用旧速度或部分roster替代。',
        '继续使用已冻结RouteA/pairs/winners，primary KL(FP||student)[8192,32768)，secondary[2048,32768)同轨迹；anchor后持续QDQ，不要求原协议没有的跨GPU/driver FP bitwise identity。',
    ]
    assessment=f'''# 隔离修复可行性（审计提案，不是修复执行）

最终判定：`{VERDICT}`。

首个实际失败已定位为 frozen worker 第63行的全局 Torch deterministic-algorithms flag 断言。GPU6先失败，GPU3随后同一门禁失败；两条stream均未进入。原始STOP/log及对应f1fa提交文件均已重哈希相符。因此当前证据支持小范围 runtime 初始化/guard配套问题的隔离修复候选，不支持把失败归为预算耗尽或P1/L7科研失效。

## 预计修改范围及风险

可提议在新worker初始化、首次CUDA模型构造/forward前显式设置确定性算法模式，并在继承Capture/seed助手前后记录模式、TF32与CUBLAS配置；保留原断言，不能删门禁或把FAIL改成PASS。具体设置位置须在新代码审查中确认。旧源码无显式enable，不等于已证明所有未展开的legacy助手都不会设置/重置；原失败直接证明assert时flag为False。

这只是有明确根因的工程候选，不能保证一行修复即可通过：installed Torch源码说明开启该模式可能切换算法或对不支持的operation抛RuntimeError，可能改变性能和数值路径。若后续要求换kernel/运行库/量化语义或弱化门禁，本候选即不再是本审计建议的小范围修复，必须停止并重新评估。

| 科学项 | 修复计划允许变化？ |
| --- | --- |
| dataset、IDs、token内容 | 不允许 |
| 16-rotation roster、tensor/权重 | 不允许 |
| frozen pairs/predicted winners | 不允许 |
| 原P1、L7 Functional/State/Total数学定义 | 不允许 |
| 每token INT8-R128 QDQ/inverse/writeback语义 | 不允许 |
| 同GPU live FP配对原则 | 不允许；新代码需新同GPUreference，不混旧不合格reference |
| endpoint32768、primary/secondary窗口、方向 | 不允许 |
| deterministic初始化/flag telemetry、新code hash | 可另行授权、隔离冻结；本轮未改 |

原L7本就H128 recurrent-aware。Phase A/B的matched-window deployment损失应用并非历史512-token训练采样完整重演，排名冗余也不是gradient等价或训练优势。本轮不重新定义P1，不训练。

## 资源和预算

已发生device-time={total:.15f} GPUh；保守release bound={release:.15f} GPUh，历史cap=16。按保守界限剩余{16-release:.15f} GPUh。两指定3090本次只读快照无compute apps/fuser holders，驱动580.178.04、kernel7.0.0-38-generic、Torch2.7.1+cu128匹配冻结字段。结果/默认临时/缓存目录当前容量充足，但没有未来资源预约。

微基准为0，完整四条32K+两FP成本与32Khybrid-cache容量仍未可验证预估。剩余额度是算术余量，不是正式长程资格或授权。修复不许直接续旧receipt/清latch，不许先跑部分有利样本。

## 下一阶段必须满足的条件（全部未在本轮执行）

'''+'\n'.join(f'{i}. {x}' for i,x in enumerate(next_conditions,1))+'''

## 授权边界

本轮在审计结果处停止。没有实施上述设置、没有forward/backward/optimizer.step、没有新FP/32K标签、没有安装/升级/清理；所有旧FAIL/STOP、模型/rotation、冻结CSV/JSON/hash manifest保持原样。`ELIGIBLE_FOR_ISOLATED_REPAIR`只表示可提交隔离工程方案，不表示GPU qualification已PASS，更不授权重启实验。

NO_NEW_GPU_SCIENTIFIC_RUN = TRUE
NO_NEW_32K_LABELS = TRUE
NO_TRAINING = TRUE
'''
    write('REPAIR_FEASIBILITY_ASSESSMENT.md',assessment)

    first=failures[0]; selected_summary=[]; ranking_tables=[]
    for entry in SEL['documents']:
        doc=entry['document']; pair=entry['pair']; ca,cb=pair['condition_a'],pair['condition_b']; vals=all_panels[(doc,'MEAN_256_1024')]
        selected_summary.append(f'| {doc} | GPU{entry["gpu"]} | {ca} vs {cb} | {entry["predicted_winners"]["P1"]} | {entry["predicted_winners"]["L7_TOTAL_ORIGINAL"]} | {pair["rank_gap_a"]:g} / {pair["rank_gap_b"]:g} |')
        lines=[f'### {doc}：两窗平均分与完整16条件midrank','', '| rotation | P1 | P1 rank | L7_TOTAL_ORIGINAL | L7 rank |','| --- | ---: | ---: | ---: | ---: |']
        rp,rl=ranks(vals['P1']),ranks(vals['L7_TOTAL_ORIGINAL'])
        lines += [f'| {c} | {vals["P1"][c]:.17g} | {rp[c]:g} | {vals["L7_TOTAL_ORIGINAL"][c]:.17g} | {rl[c]:g} |' for c in conditions]
        ranking_tables.append('\n'.join(lines))
    significant=['PROSPECTIVE_DOCUMENT_AND_PAIR_SELECTION.json','PROSPECTIVE_DOCUMENT_SELECTION.json','PHASE_B_PAIR_METHODS_FREEZE.json',
        'PHASE_B_CALIBRATION_MATCHED_TABLES.json','SOURCE_IDENTITIES.json','PHASE_B_CALIBRATION_CODE_FREEZE.json',
        'PHASE_B_CALIBRATION_CONFIG.json','PERSISTENT_CODE_AND_ASSET_FREEZE_V1.json','run_persistent_stage_v1.py',
        'persistent_engine_v1.py','persistent_stage_common_v1.py','final_stop_20261011/DELIVERABLE_MANIFEST.json',
        'final_stop_20261011/RESULT_AND_DELIVERY_HASHES.json','final_stop_20261011/SHA256SUMS.txt',
        'final_stop_20261011/GPU_COST_LEDGER.json','GPU_COST_LEDGER.json']
    hash_lines=[f'| `{n}` | `{identity(n)["sha256"]}` | MATCH |' for n in significant]
    for f in failures:
        for key in ('original_stop','original_log'):
            x=f[key]; hash_lines.append(f'| `{x["path"]}` | `{x["sha256"]}` | VERIFIED |')
    resource_time=stamp(E['resources']['observed_epoch'])
    report=f'''# Ling/KDA Phase B STOP 只读取证审计

最终判定：`{VERDICT}`。这是隔离工程修复候选，**不是新GPU实验许可，也不是长期科研确认**。本轮先制定审计计划，随后读取真实服务器资产、静态源码与git记录，最后在全新目录交付；未执行修复。

## 1. 冻结科学选择恢复与资格

| 本轮新文档 | 固定设备 | frozen pair | P1 predicted winner | L7_TOTAL predicted winner | 双方midrank gap |
| --- | --- | --- | --- | --- | --- |
'''+ '\n'.join(selected_summary)+f'''

A历史Phase A=37702/5396；B既有P1-vs-B0=48693/51410；C本次真正冻结Phase B=860/5956，互不混用。冻结输入与prior path registry明确排除A/B及其他旧文档。文档内容raw/token SHA均重验MATCH，合法token数各32769。数据身份为本地`phase3_mixed_hardware_v1_20261003` PG19 test快照路径+内容SHA；未在freeze发现上游dataset revision，不以目录日期虚构Hub版本。旧exact-tokenizer-reencoding证据已重哈希；本轮未重tokenize。

Route A保持P1 vs原L7_TOTAL。窗口起点256/1024、各H128，对应[256,384)、[1024,1152)，全18KDA/all16heads；两窗等权平均、lower safer。完整16 roster未改，每个rotation身份见selection audit。原L7本就H128 recurrent-aware；原helper定义保留，matched deployment tensor应用并非历史512-token训练采样完整重演。P1采用已冻结query-readout energy定义，本轮没有重新发明指标。

独立CPU检查：{row_checks}个原score cell逐一与既有两篇canonical匹配；{rank_checks}个midrank cell独立重算MATCH；两文档全部eligible pair列表及固定min-gap/lexicographic优先顺序完全匹配原freeze，未重新执行selector或发布新selection。原始raw分数、每窗和汇总完整排序、FP tensor身份、roster/source身份保留在`PHASEB_FROZEN_SELECTION_AUDIT.json`。

'''+ '\n\n'.join(ranking_tables)+f'''

## 2. Outcome leakage审计

结论：`NO_OUTCOME_LEAKAGE_FOUND_IN_AUDITED_ASSETS`。pair methods冻结于{stamp(D['PHASE_B_PAIR_METHODS_FREEZE.json']['epoch'])}，早于正式calibration；selection于{stamp(SEL['epoch'])}，晚于两篇完整calibration且早于persistent launch。selector静态数据入口只有短calibration、rules/route、input/source身份及账本，不读新长程标签；文档hash order/seed0及prior exclusions一致。

两qualification在stream前失败，`runs_b_persistent`无canonical/micro/formal结果，完整cost gate不存在，新增32K FP/labels/science units均0。这与freeze声明new_long_outcomes_generated_or_read=false一致。该结论限于审计资产，不能声称证明所有未记录的人为访问历史；不使用既有B组长程KL来选择C组pair。

## 3. 首个实际qualification failure，不按终态猜测

首失败：`{first['gate']}`，GPU6/860，worker第63行：

```python
assert torch.are_deterministic_algorithms_enabled()
```

预期True；实际False由原失败的boolean assert与installed getter源码推定，原日志未独立打印flag值。STARTED={first['started_time']}，STOP={first['stopped_time']}。对应原STOP：`{first['original_stop']['path']}`；SHA=`{first['original_stop']['sha256']}`。

```text
'''+first['original_traceback'].strip()+f'''
```

GPU3/5956随后于{failures[1]['stopped_time']}同一行失败。两log都显示32/32模型shard完成加载，随后同一traceback；model-type/deprecation/generation-flags警告不是首个FAIL。实际加载完成不等于执行了模型forward。

顺序：CPU freeze/input/device/900s完整资格roster reservation→LAUNCH_AUTHORIZATION→worker verify/receipt/deadline→STARTED→GPU runtime/UUID独占assert→peer/deadline guard→load assets/inputs→Capture模型加载→globalflag FAIL→exclusive STOP→共享latch。各中间门禁只可由原控制流证明已返回，缺少独立精确timestamp；本报告不伪造逐gate时间或把它们写成完整GPU资格PASS。原记录的细节、两卡事件顺序、source lines、原traceback见`FIRST_FAILURE_GATE_REPORT.json`。

TF32/CUBLAS后续assert及第一token尚未到达；真实512-token replay/full-logit parity、persistent QDQ/writeback消费、hybrid cache数值、finite/repeat、8448micro/32K都未执行。此前Phase A/B短程GPU PASS仍是原阶段证据，不替代新persistent资格。晚于首STOP的guard exception是已存在latch所致，不是首因；两.partial latch保留；预GPU CPU构建错误也单独保留，不混为这次首个qualification门禁。

分类：工程runtime配置/guard配套前置条件不匹配。没有证据把它归为测得的数值不确定性、QDQ科学失败、OOM、硬件异常或预算预测失败。worker无显式enable后又要求global flag为True，固定FLA selector/seed/CUBLAS配置并不等价于这个flag。可提出小范围显式runtime初始化与telemetry修复，但本轮不改源码、不移除assert、不清STOP。

## 4. 当前资源、已发生开销及预算边界

只读快照时间={resource_time}。GPU3=`{idle['3']['uuid']}`；GPU6=`{idle['6']['uuid']}`。两张RTX3090各24576MiB，当前15MiB/0%util，nvidia compute-apps为空且fuser无holder。旧PID341582/341594及对应tmux已退出。空闲是时间点事实，不是独占预约。

当前kernel=`{E['resources']['kernel']}`；driver=`{idle['3']['driver']}`；Torch=`{E['resources']['packages']['torch']}`；CUDA runtime distribution=`{E['resources']['packages']['nvidia-cuda-runtime-cu12']}`；fla-core=`{E['resources']['packages']['fla-core']}`；Transformers=`{E['resources']['packages']['transformers']}`；Triton=`{E['resources']['packages']['triton']}`。版本来自CPU metadata/文件及nvidia，不importGPU库；当前Torch compile/runtime通过+cu128/metadata间接核对，未做forward资格。所有冻结依赖SHA匹配不等于未绑定runtime文件全体已资格。reboot-required/shutdown-schedule均无标记，仅为快照，不保证未来无维护。

结果分区free={E['resources']['disks'][ROOT]['free_bytes']} bytes（{E['resources']['disks'][ROOT]['free_bytes']/1024**3:.3f}GiB）；/tmp={E['resources']['disks']['/tmp']['free_bytes']/1024**3:.3f}GiB；Triton/Torch缓存同结果分区；/dev/shm={E['resources']['disks']['/dev/shm']['free_bytes']/1024**3:.3f}GiB。`/scratch`和`/data/zypan/scratch`不存在，不把它们当已预留scratch。原2GiB结果空间门槛现在满足，未来峰值temp/cache/storage/连续性仍需测量和预检；没有清理缓存。

10条历史device job（A四条、B smoke+calibration四条、persistent qualification两条）由各receipt到canonical完成或STOP时间独立重算，总device-time={total:.15f}GPUh，process-release保守界限={release:.15f}GPUh。剩余分别{16-total:.15f}/{16-release:.15f}GPUh；不是预算耗尽。初始0账本仅保留，不能替代总A+B+persistent账本；等待、其他实验、CPU不计本轮GPU开销。

两qualification各900s cap总0.5GPUh只是限额，不是保证成功或未来成本。真实8448micro=0，因此完整四student32K+两必要同GPU FP保守cost/VRAM还不能验证预估。原冻结规则需两卡micro实际最慢post2048block速率×32768/8448增长包络、1.5margin+30s、加载/FP/两student/所有历史资格计入；native-cache extrapolated peak<=85%显存，完整总<=16。不可用其他runner历史速度或partial sample替代。详见`RESOURCE_AND_BUDGET_AUDIT.json`。

## 5. 可修复性与后续资格

结论`{VERDICT}`仅表示具备明确原始首因、选择有效且身份未变的隔离工程候选。显式enable可能切换algorithm/触发unsupported-op RuntimeError并改变性能；不能宣称修复一定成功，也不能把本CPU审计冒充GPU PASS。修复预计只涉及runtime setup/observability与新code/freeze，dataset/roster/pairs/winners/P1/L7/QDQ/FP同GPU配对原则/metric/endpoint均不允许变。

下一步条件（本轮全部未执行）：

'''+ '\n'.join(f'{i}. {x}' for i,x in enumerate(next_conditions,1))+f'''

## 6. Provenance及字节保护

真实主worktree：`{ROOT}`。当前HEAD=`{E['git']['head']['stdout'].strip()}`，不是测量commit；历史PhaseA/CPUanalyze/calibration/pairaudit/persistent五提交均实际存在。persistent14测量文件逐个git-show内容SHA与runfile相等，persistent commit=`f1fa49883d5f1f65ed47d8ddc0b37a62cbee49d1`。当前tracked staged/unstaged diff为空，但有{len(E['git']['status_before']['stdout'].splitlines())}条untracked evidence，所以不声称整个worktree无文件变化。两上游source仓库HEAD均与SOURCE_IDENTITIES相同，bridge仓库另有既有dirty/untracked项目，完整status见provenance；绑定科学源码SHA全部匹配。

服务器{len(E['hash_checks'])}项manifest/sidecar/source/asset检查全部MATCH；32权重分片完整SHA重算且metadata与fresh audit相符；旧80canonical+80sidecar及两永久Attempt STOP重哈希MATCH。没有独立128panel覆盖证据，不虚报128项PASS。原实验335个非.git/非__pycache__文件审计前后及补充核验字节相同；本轮仅新审计目录产生文件，不修改模型/rotations/旧reports/hashmanifest。

| 原始相对路径（STOP/log为绝对路径） | 实际SHA256 | 审计 |
| --- | --- | --- |
'''+ '\n'.join(hash_lines)+f'''

完整原证据/command stdout/原git status/所有file identity/hash-check结果在`READONLY_SERVER_EVIDENCE.json`，SHA=`{sha(OUT/'READONLY_SERVER_EVIDENCE.json')}`；补充静态依赖/原receipts/进程观察在`READONLY_SERVER_SUPPLEMENT.json`，SHA=`{sha(OUT/'READONLY_SERVER_SUPPLEMENT.json')}`。没有使用用户交接文本模拟远端资产。三个JSON各附完整provenance；新deliverable自身hash见`SHA256SUMS.txt`（manifest正常不hash自身）。

## 7. 精确交付路径和本轮停止点

本地：`{str(OUT)}`

远端：`{E['output_root']}`

六项指定输出为PHASEB_FROZEN_SELECTION_AUDIT.json、FIRST_FAILURE_GATE_REPORT.json、RESOURCE_AND_BUDGET_AUDIT.json、REPAIR_FEASIBILITY_ASSESSMENT.md、FINAL_READONLY_AUDIT_REPORT.md、SHA256SUMS.txt；另保留stdlib-only collector/builder和真实只读证据用于复核。不对旧实验追加PASS/新STOP或更改monitor。本轮结果交付即停止，不开展提案中的修复或后续资格。

NO_NEW_GPU_SCIENTIFIC_RUN = TRUE
NO_NEW_32K_LABELS = TRUE
NO_TRAINING = TRUE
'''
    write('FINAL_READONLY_AUDIT_REPORT.md',report)
    items=sorted(p for p in OUT.iterdir() if p.is_file() and p.name!='SHA256SUMS.txt')
    write('SHA256SUMS.txt','\n'.join(f'{sha(p)}  {p.name}' for p in items))
    print(json.dumps(dict(verdict=VERDICT,score_cells_verified=row_checks,rank_cells_verified=rank_checks,
        total_device_gpu_hours=total,release_bound=release,remaining_conservative=16-release,
        six_requested_deliverables_generated=True,output=str(OUT),**FLAGS)))

if __name__=='__main__': main()
