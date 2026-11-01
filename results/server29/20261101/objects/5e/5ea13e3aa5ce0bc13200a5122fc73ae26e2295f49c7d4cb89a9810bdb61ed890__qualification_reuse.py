"""Exact read-only reuse of the sole BYTEVIEW48 origin; never foreign dispatch.

Old receipts remain historical and are read from their fixed original55 root.
No old STARTED, RESULT or interval is created in this execution. The failed
49th old worker is not a qualification, even though its RESULT says PASS.
"""
from pathlib import Path
import execution_control as C
from control_plan import build_qualification_plan

ORIGIN_ROOT='/tmp/QWEN_GDN_SYM8_G32_WSCALE_ROWBAL_AIME26_SEED1_V1_BYTEVIEW_REPAIR_20261025'
ORIGIN_SOURCE_SHA256='05901486d72e6f62aba67e61961d6d62043dabef14340b46034a072a004ce5e1'
ORIGIN_PERMIT_SHA256='3b03a13eafac5d8997b141f5d099b39f682aee10a1a57bedc79d33dc362c611c'
ORIGIN_AUDIT_SHA256='371ec55ebab83fe9059aab0092e880e7eb52dac6bc6370401665231c621b3902'
ORIGIN_FINAL_REVIEW_SHA256='9b82cc4160af16f46ecd9fd3a5438178f59d3d0a54235a9e21ad3dfe3a336482'
ORIGIN_TRITON_SHA256='bc695cc73e549272d9231cfe1f6b833dbf3519c9964e3820746acfee92dcafe5'
HUMAN_AUTHORIZATION_SHA256='b62658907736235803dacb60b73af292ce8f07025389726f46cf8961147c56f4'
AUDIT_PATH='provenance/REUSED_QUALIFICATION48/CPU_ORIGIN48_READONLY_AUDIT.json'
REVIEW_PATH='provenance/REUSED_QUALIFICATION48/FINAL_INDEPENDENT_TERMINAL_REVIEW.json'
FRESH_PATH='analysis/REUSE48_DUAL_HOST_FRESH_REVIEW.json'
REMAINING_IDS=tuple('GPU1_'+m+'_MATCHED4096' for m in ('base','W','R','RW'))

def _safe_file(root,relative):
    rel=Path(relative)
    C.require(not rel.is_absolute() and '..' not in rel.parts and str(rel)==relative,'Unsafe provenance relative path')
    p=Path(root)/rel
    C.require(p.is_file() and not p.is_symlink(),'Fixed provenance file missing or symlink:'+relative)
    return p

def _bound_file(root,row):
    p=_safe_file(root,row['path'])
    C.require(p.stat().st_size==row['bytes'] and C.sha(p)==row['sha256'],'Fixed provenance SHA/bytes differ:'+row['path'])
    return p

def remaining_units(plan):
    expected=build_qualification_plan(C.UUIDS,C.AUTHORIZED_GPUS)['qualification_units']
    C.require(plan['qualification_units']==expected,'Original exact52 mathematical qualification roster differs')
    units=[u for u in expected if u['id'] in REMAINING_IDS]
    C.require(tuple(u['id'] for u in units)==REMAINING_IDS and all(u['gpu']==1 and u['kind']=='matched_performance' for u in units),'Only original four matchedGPU1 units may be new qualifications')
    return units

def reused_units(plan):
    remaining_units(plan)
    units=[u for u in plan['qualification_units'] if u['id'] not in REMAINING_IDS]
    C.require(len(units)==48 and all(sum(u['gpu']==g for u in units)==16 for g in C.AUTHORIZED_GPUS),'Reuse roster must be exact16 per original UUID')
    return units

def require_current_dispatch(root,unit,formal=False):
    """Admission before Popen and worker STARTED; not a reuse/proof constructor."""
    root=Path(root);plan=C.read(root/'GPU_EXECUTION_PLAN.json')
    allowed=C.read(root/'FORMAL_JOB_MANIFEST.json')['jobs'] if formal else remaining_units(plan)
    if formal:C.validate_jobs(root,allowed)
    uid=unit if isinstance(unit,str) else unit['id'];matches=[u for u in allowed if u['id']==uid]
    C.require(len(matches)==1 and (isinstance(unit,str) or matches[0]==unit),'Reused/foreign/rewritten unit cannot dispatch in the new execution')
    return matches[0]

def assert_no_legacy_current_evidence(root,plan):
    for unit in reused_units(plan):
        C.require(not (Path(root)/'execution'/unit['id']).exists(),'Historical qualification must not appear in current execution: '+unit['id'])

def admit_reused48(root,plan):
    """Fresh SHA/bytes and original field gates, only one hard-coded old origin."""
    from control_qualification import inspect_unit
    root=Path(root);units=reused_units(plan);assert_no_legacy_current_evidence(root,plan)
    admission=C.read(root/'QUALIFICATION_REUSE_ADMISSION.json')
    C.require(admission['status']=='PASS_REUSE48_FIXED_BYTEVIEW_ORIGIN' and admission['qualification_reused']==48 and admission['qualification_new']==4 and admission['formal_reused']==0 and admission['formal_new']==90 and admission['old49_failed_exit_not_reused'] is True and admission['no_current_legacy_started_result_interval'] is True,'Exact48 reuse and4fresh/90fresh admission scope differs')
    C.require(admission['origin_audit_path']==AUDIT_PATH and admission['origin_audit_sha256']==ORIGIN_AUDIT_SHA256 and admission['origin_final_review_path']==REVIEW_PATH and admission['origin_final_review_sha256']==ORIGIN_FINAL_REVIEW_SHA256,'Only the fixed independent BYTEVIEW origin proofs are permitted')
    C.require(admission['human_authorization_sha256']==HUMAN_AUTHORIZATION_SHA256==C.sha(root/'operations/HUMAN_REUSE48_ARCHIVE29_AUTHORIZATION.json'),'New one-attempt human reuse grant differs')
    C.require(C.sha(_safe_file(root,AUDIT_PATH))==ORIGIN_AUDIT_SHA256 and C.sha(_safe_file(root,REVIEW_PATH))==ORIGIN_FINAL_REVIEW_SHA256,'Original independent audit bytes differ')
    audit=C.read(root/AUDIT_PATH);review=C.read(root/REVIEW_PATH)
    C.require(audit['status']=='PASS_CPU_READONLY_48_LEGAL_ORIGIN_UNITS' and audit['origin_execution_root']==ORIGIN_ROOT and audit['qualification_reused_count']==48 and audit['formal_credit']==0 and audit['original_stop_preserved'] is True,'Original origin identity/real48/legal credit differs')
    C.require(review['status']=='PASS_FINAL_INDEPENDENT_TERMINAL_DELIVERY_REVIEW' and review['qualification_PASS']==48 and review['formal_complete']==0 and review['source_count']==334 and review['exact_owned_processes_alive']==0,'Original STOP independent completion identity differs')
    failed=audit['old49th'];C.require(failed['unit']==REMAINING_IDS[0] and failed['actual_exit']==-11 and failed['reusable'] is False,'Old49 nonzero exit must remain excluded')
    C.require(admission['dual_host_fresh_review_path']==FRESH_PATH and isinstance(admission['dual_host_fresh_review_sha256'],str) and len(admission['dual_host_fresh_review_sha256'])==64,'Fresh dualhost origin/source/raw admission missing')
    freshfile=_safe_file(root,FRESH_PATH);C.require(C.sha(freshfile)==admission['dual_host_fresh_review_sha256'],'Fresh dualhost origin review SHA differs');fresh=C.read(freshfile)
    C.require(fresh['status']=='PASS_FRESH_DUAL_HOST_REUSE48_BYTEVIEW_SOURCE_RAW' and set(fresh['hosts'])=={'55','29'},'Fresh two-host origin provenance review missing')
    for host in ('55','29'):
        h=fresh['hosts'][host]
        C.require(h['status']=='PASS' and h['source_count']==334 and h['unit_count']==48 and h['source_sha256']==ORIGIN_SOURCE_SHA256 and h['proof_files_fresh_SHAbytes']==555 and h['science_count']==5397 and h['science_bytes']==1731153792,'Fresh host source/proof48/555 and complete5397 admission incomplete:'+host)
    origin=Path(ORIGIN_ROOT)
    C.require(C.sha(origin/'SOURCE_FREEZE.json')==ORIGIN_SOURCE_SHA256 and C.sha(origin/'GPU_PERMIT.json')==ORIGIN_PERMIT_SHA256,'Fixed original source/permit differs')
    oldsource=C.read(origin/'SOURCE_FREEZE.json');C.require(len(oldsource['files'])==334,'Original frozen source roster differs')
    for source in oldsource['files']:_bound_file(origin,source)
    for name,binding in audit['origin_controls'].items():_bound_file(origin,dict(path=name,**binding))
    C.require(C.sha(origin/'analysis/TRITON_RUNTIME_FRESH_SHA.json')==ORIGIN_TRITON_SHA256,'Original Triton runtime binding differs')
    originalpins={x['path']:x for x in oldsource['files']};math=admission['source_math_unchanged'];C.require(len(math)>=10,'Unchanged mathematics/protocol source pins missing')
    required={'AIME26_SEED1_PROTOCOL_FREEZE.json', 'src/runtime_engine.py', 'src/unified_adapter.py', 'protocol/aime26_scorer_v4.py', 'CALIBRATION_REUSE_ADMISSION.json', 'CALIBRATION_PARAMETER_SELECTION.json', 'TRANSPORTED_QUERY_OMEGA.json', 'AIME26_SEED1_PROTOCOL.json', 'src/unified_codec.py', 'protocol/sampler.py', 'src/four_codec_adapter.py', 'protocol/aime26_frozen.jsonl', 'src/non_aime_diagnostics.py', 'src/record_validation.py', 'FORMAL_JOB_MANIFEST.json', 'src/control_plan.py', 'protocol/FREE_CANARY_INPUT.json', 'AIME26_DATASET_FREEZE.json', 'src/triton_runtime_guard.py', 'protocol/non_aime_fixture.json', 'protocol/common.py', 'GPU_EXECUTION_PLAN.json', 'src/protocol_binding.py', 'src/gpu_codec_fields.py', 'QUESTION_GPU_ASSIGNMENT.json', 'CALIBRATION_DATA_FREEZE.json'}
    C.require(required.issubset({x['path'] for x in math}),'Core math/native/prompt/sampler/scorer identity pins missing')
    for binding in math:
        C.require(binding==originalpins.get(binding['path']),'Source math/protocol pin is not original frozen source')
        _bound_file(root,binding)
    byid={x['unit']:x for x in audit['units']};C.require(len(byid)==48 and list(byid)==[u['id'] for u in units],'Fixed reused48 order/IDs differ')
    rows=[];proofs=0
    for unit in units:
        bound=byid[unit['id']];C.require(bound['original_definition']==unit and bound['actualexit0'] is True and bound['release']=='PASS','Fixed reused qualification definition/exit/release differs')
        for f in bound['files']:
            C.require(f['path'].startswith('execution/'+unit['id']+'/'),'Provenance escaped its exact original unit')
            _bound_file(origin,f);proofs+=1
        row=inspect_unit(origin,unit,source_sha256=ORIGIN_SOURCE_SHA256,permit_sha256=ORIGIN_PERMIT_SHA256,triton_manifest_sha256=ORIGIN_TRITON_SHA256,triton_task_root=ORIGIN_ROOT)
        w=C.read(origin/'execution'/unit['id']/'RESULT.json');expected=[ '/data/ydai/miniconda3/envs/bitdecode/bin/python','-u','-B',ORIGIN_ROOT+'/scripts/gpu_unit_worker.py','--root',ORIGIN_ROOT,'--unit',unit['id'] ]
        C.require(w['identity']==bound['identity'] and w['identity']['cmdline']==expected,'Original actual argv/starttick identity binding differs')
        C.require(row['result_sha256']==bound['result_sha256'] and row['resource_interval_sha256']==bound['resource_interval_sha256'],'Original result/actual interval hashes differ')
        row.update(evidence_origin='REUSED_BYTEVIEW_20261025',origin_root=ORIGIN_ROOT,origin_source_sha256=ORIGIN_SOURCE_SHA256,origin_permit_sha256=ORIGIN_PERMIT_SHA256);rows.append(row)
    C.require(proofs==543 and len(audit['origin_controls'])==12,'Exact original durable unit/control proof counts differ')
    return dict(status='PASS_REUSE48_EXACT_BYTEVIEW_ORIGIN_EXIT0_RELEASE',count=48,units=rows,origin_root=ORIGIN_ROOT,origin_source_sha256=ORIGIN_SOURCE_SHA256,origin_permit_sha256=ORIGIN_PERMIT_SHA256,origin_final_review_sha256=ORIGIN_FINAL_REVIEW_SHA256,admission_sha256=C.sha(root/'QUALIFICATION_REUSE_ADMISSION.json'),old49_credit=0,new_GPU_intervals=0,new_formal_credit=0,proof_files_fresh_SHAbytes=proofs+len(audit['origin_controls']),unit_proof_files_fresh_SHAbytes=proofs,origin_control_files_fresh_SHAbytes=len(audit['origin_controls']))
