"""Meaningful CPU-only failure/evidence/order tests; no model/GPU operations."""
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types

SCRIPT=Path(__file__).resolve().parent;ROOT=SCRIPT.parent
sys.path.insert(0,str(SCRIPT))
import library_identity as L
spec=importlib.util.spec_from_file_location('repair_worker_cpu',SCRIPT/'cublas_worker.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
checks=[]
def check(name,value):
    if not value:raise AssertionError(name)
    checks.append(dict(name=name,status='PASS'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mapping(p,*,deleted=False,inode_delta=0):
    s=p.stat()
    return f'1000-2000 r-xp 00000000 {os.major(s.st_dev):x}:{os.minor(s.st_dev):x} {s.st_ino+inode_delta} {p}'+(' (deleted)'if deleted else '')+'\n'
def binding(p):return dict(path=str(p),resolved_path=str(p),bytes=p.stat().st_size,sha256=sha(p),role=L.classify_library(p))
def expect_failure(contract,phase,maps,stage='PREFLIGHT'):
    try:L.library_identity_gate(contract,phase,stage,maps_reader=lambda:maps)
    except L.LibraryIdentityError as exc:return exc
    raise AssertionError('Expected identity gate failure')

ROOT.joinpath('qualification/library_identity_cpu_tests').mkdir(parents=True,exist_ok=True)
with tempfile.TemporaryDirectory(prefix='synthetic_',dir=ROOT/'qualification/library_identity_cpu_tests')as td:
    tmp=Path(td);files=tmp/'files';files.mkdir()
    torch=files/'libtorch_cpu.so';torch.write_bytes(b'CPU dummy torch bytes')
    cublas=files/'libcublas.so.12';cublas.write_bytes(b'CPU dummy cuBLAS bytes')
    audio=files/'libtorchaudio.so';audio.write_bytes(b'CPU dummy torchaudio bytes')
    extra=files/'libtorch_extra.so';extra.write_bytes(b'CPU dummy extra bytes')
    cudnn=files/'libcudnn.so.9';cudnn.write_bytes(b'CPU dummy cuDNN bytes')
    ordinary=files/'ordinary_data.bin';ordinary.write_bytes(b'Unselected map must be retained')
    known=dict(prospective_library_bindings=[binding(torch),binding(cublas)])
    allmaps=mapping(torch)+mapping(cublas)+mapping(audio)+mapping(ordinary)
    failure=expect_failure(known,tmp/'missing_audio',allmaps)
    check('unbound_lib_torchaudio_is_selected_torch_and_rejected',L.classify_library(audio)=='torch'and failure.report['missing_bindings']==[str(audio)])
    check('complete_nonselected_map_record_retained',any(r.get('path')==str(ordinary)for r in failure.report['all_mapped_paths']))
    check('complete_raw_maps_preserved_before_raise',(tmp/'missing_audio/library_identity/PREFLIGHT_MAPS.txt').read_text()==allmaps)
    identity=tmp/'missing_audio/library_identity/PREFLIGHT_IDENTITY.json'
    check('failed_identity_report_durable_before_raise',identity.is_file()and json.loads(identity.read_text())['status']=='FAIL')
    check('failure_summary_receipt_bytes_SHA_bind_report',sha(identity)==failure.summary['identity_receipt']['sha256']and identity.stat().st_size==failure.summary['identity_receipt']['bytes'])
    expected=dict(prospective_library_bindings=known['prospective_library_bindings']+[binding(audio)])
    good=L.library_identity_gate(expected,tmp/'all_bound','PREFLIGHT',maps_reader=lambda:allmaps)
    check('bound_audio_no_prefix_exclusion_required',good['status']=='PASS'and len(good['identities'])==3)
    check('semantic_digest_omits_memory_addresses_PID_timestamps',good['identity_digest']==L.digest(L.canonical(good['identities'])))
    altered=allmaps.replace('1000-2000','3000-4000')
    other=L.library_identity_gate(expected,tmp/'other_addresses','PREFLIGHT',maps_reader=lambda:altered)
    check('same_semantic_library_identity_despite_map_address_changes',other['identity_digest']==good['identity_digest'])
    mismatch=binding(cudnn);mismatch['bytes']+=1;mismatch['sha256']='0'*64
    multiple=dict(prospective_library_bindings=known['prospective_library_bindings']+[mismatch])
    multi=expect_failure(multiple,tmp/'multiple',allmaps+mapping(extra)+mapping(cudnn))
    check('all_missing_bindings_collected_without_short_circuit',multi.report['missing_bindings']==sorted([str(audio),str(extra)]))
    check('both_bytes_and_SHA_mismatch_collected',multi.report['mismatches'][0]['fields']==['bytes','sha256'])
    check('full_later_library_detail_retained_after_first_missing',len(multi.report['selected_library_details'])==5)
    inode=expect_failure(known,tmp/'wrong_inode',mapping(torch,inode_delta=1)+mapping(cublas))
    check('mapped_inode_mismatch_rejects_current_path_hash_substitution',any(x['kind']=='MAPPED_FILE_DEVICE_INODE_MISMATCH'for x in inode.report['issues']))
    deleted=expect_failure(expected,tmp/'deleted',mapping(torch)+mapping(cublas)+mapping(audio,deleted=True))
    check('deleted_selected_map_not_silently_excluded',any(x['kind']=='DELETED_MAPPED_LIBRARY'for x in deleted.report['issues']))
    lazy=L.library_identity_gate(dict(prospective_library_bindings=[binding(torch)]),tmp/'lazy_pre','PREFLIGHT',maps_reader=lambda:mapping(torch))
    check('preflight_allows_lazy_cuBLAS_without_forward_probe',lazy['status']=='PASS')
    absent=expect_failure(dict(prospective_library_bindings=[binding(torch)]),tmp/'lazy_post',mapping(torch),'POSTFLIGHT')
    check('postflight_preserves_original_cuBLAS_requirement',any(x.get('role')=='cuBLAS'and x['kind']=='REQUIRED_RUNTIME_ROLE_ABSENT'for x in absent.report['issues']))
    snapshots=iter([mapping(torch)+mapping(cublas),allmaps+mapping(extra)])
    try:L.library_identity_gate(known,tmp/'concurrent_map_change','PREFLIGHT',maps_reader=lambda:next(snapshots))
    except L.LibraryIdentityError as exc:concurrent=exc
    check('library_appearing_during_gate_rejected_and_missing_list_complete',concurrent.report['missing_bindings']==sorted([str(audio),str(extra)])and any(x['kind']=='SELECTED_MAP_IDENTITY_CHANGED_DURING_GATE'for x in concurrent.report['issues']))
    def unreadable():raise PermissionError('synthetic maps denied')
    try:L.library_identity_gate(known,tmp/'unreadable_maps','PREFLIGHT',maps_reader=unreadable)
    except L.LibraryIdentityError as exc:unread=exc
    check('maps_read_failure_retains_diagnostic_report',len(unread.report['maps_read_errors'])==2 and (tmp/'unreadable_maps/library_identity/PREFLIGHT_IDENTITY.json').is_file())
    try:L.library_identity_gate(expected,tmp/'all_bound','PREFLIGHT',maps_reader=lambda:allmaps)
    except FileExistsError:duplicate=True
    else:duplicate=False
    check('immutable_gate_evidence_cannot_be_overwritten',duplicate)
    # Integration: execute real worker main with CPU dummy admission/runtime.
    # A failing preflight must prevent any qualify dispatch (thus first forward).
    saved={name:getattr(worker,name)for name in ['admission','runtime','qualify','loaded_library_identity','mark_progress']}
    oldargv=sys.argv[:]
    def integrate(name,pre_maps,post_maps=None):
        case=tmp/name;phase=case/'execution/B0_FP_Q0';phase.mkdir(parents=True)
        auth=dict(device=dict(physical_index=0,uuid='GPU-SYNTHETIC-NOT-REAL'),shared_control_root=str(case))
        unit=dict(unit_id='B0_FP_Q0',repeat=0)
        contract=dict(known,source_bindings=dict(core=dict(sha256='0'*64)))
        seen=[]
        worker.admission=lambda root,args:(auth,contract,list(range(256)),unit,phase)
        worker.runtime=lambda *args:(types.SimpleNamespace(_core_AST_sha256='CPU-SYNTHETIC'),None,{'CURRENT':{}},{},object(),object(),{})
        def fake_qualify(*args):seen.append('SYNTHETIC_QUALIFY_DISPATCH');return {}
        worker.qualify=fake_qualify;worker.mark_progress=lambda *args,**kwargs:None
        worker.loaded_library_identity=lambda contract,phase,stage:L.library_identity_gate(contract,phase,stage,maps_reader=lambda:pre_maps if stage=='PREFLIGHT'else post_maps)
        sys.argv=['cublas_worker.py','--root',str(case),'--authorization',str(case/'AUTHORIZATION.json'),'--unit-id','B0_FP_Q0','--mode','qualify','--condition','B0','--codec','FP']
        with contextlib.redirect_stderr(io.StringIO()):
            try:worker.main()
            except SystemExit as exc:check(name+'_main_failed_closed_exit2',exc.code==2)
        return seen,json.loads((phase/'RESULT.json').read_text()),phase
    try:
        seen,result,phase=integrate('pre_fail_no_forward',allmaps)
        check('unbound_preflight_prevents_first_scientific_dispatch',seen==[])
        check('preflight_failure_RESULT_binds_durable_evidence',result['runtime_library_preflight']['status']=='FAIL'and result['runtime_library_postflight']is None and (phase/'library_identity/PREFLIGHT_IDENTITY.json').is_file())
        seen,result,phase=integrate('late_load_post_fail',mapping(torch)+mapping(cublas),allmaps)
        check('late_loaded_unbound_library_fails_postflight',len(seen)==1 and result['runtime_library_preflight']['status']=='PASS'and result['runtime_library_postflight']['status']=='FAIL')
        check('late_load_both_stage_evidence_retained',all((phase/f'library_identity/{s}_IDENTITY.json').is_file()for s in ('PREFLIGHT','POSTFLIGHT')))
    finally:
        for name,value in saved.items():setattr(worker,name,value)
        sys.argv=oldargv

old=ROOT.parent/'QWEN_GDN_FAST_RUNTIME_QUALIFICATION_GPU_20261019_V1/staging/payload_v1/gpu3/scripts/cublas_worker.py'
oldtree=ast.parse(old.read_text());newtree=ast.parse((SCRIPT/'cublas_worker.py').read_text())
definition=lambda tree,name:next(x for x in tree.body if getattr(x,'name',None)==name)
dump=lambda node:ast.dump(node,include_attributes=False)
for name in ('transform_core','profile_replay_ast','extract','runtime','DeepRecorder','qualify','tensor_bytes','replay_progress'):
    check(name+'_AST_exact_previous_frozen_worker',dump(definition(oldtree,name))==dump(definition(newtree,name)))
def scientific_mode_branch(tree):
    main=definition(tree,'main');body=next(x for x in main.body if isinstance(x,ast.Try)).body
    return next(x for x in body if isinstance(x,ast.If)and ast.unparse(x.test)=="args.mode == 'qualify'")
check('all_scientific_mode_dispatch_branch_AST_unchanged',dump(scientific_mode_branch(oldtree))==dump(scientific_mode_branch(newtree)))
check('worker_RUN_repair_identity',worker.RUN=='QWEN_GDN_FAST_RUNTIME_QUALIFICATION_REPAIR_20261019_V1')
check('CPU_tests_never_imported_torch_transformers',not any(x=='torch'or x.startswith('torch.')or x=='transformers'or x.startswith('transformers.')for x in sys.modules))
out=dict(status='CPU_PASS_NOT_GPU_QUALIFICATION',checks=checks,GPU_imports=False,GPU_queries=0,GPU_dispatches=0,
         worker_SHA256=sha(SCRIPT/'cublas_worker.py'),library_identity_SHA256=sha(SCRIPT/'library_identity.py'),
         previous_frozen_worker_SHA256=sha(old),tests_SHA256=sha(__file__),
         source_math_AST_unchanged=True,old_sealed_root_modified=False,
         scope='Synthetic CPU files/maps and mocked worker control path; no model forward, no real runtime library preload proof.',
         runtime_library_schema=L.SCHEMA)
target=ROOT/'qualification/LIBRARY_IDENTITY_REPAIR_CPU_TESTS.json';target.parent.mkdir(parents=True,exist_ok=True)
target.write_text(json.dumps(out,sort_keys=True,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(status=out['status'],checks=len(checks),receipt=str(target),sha256=sha(target))))
