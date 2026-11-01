"""CPU-only collector entry/permit checks using a labelled temporary fixture."""
import hashlib
import json
from pathlib import Path
import tempfile
import types
import os


def run(collector_source):
    checks=[]
    with tempfile.TemporaryDirectory(prefix='WSCALE_CPU_CALIBRATION_ENTRY_ONLY_') as directory:
        root=Path(directory)/'QWEN_GDN_SYM8_G32_WSCALE_ROWBAL_AIME26_SEED1_V1'
        (root/'calibration').mkdir(parents=True);(root/'analysis').mkdir();(root/'operations').mkdir()
        mod=types.ModuleType('collector_cpu_fixture');mod.__file__=str(root/'calibration/collect_fp_once.py')
        exec(compile(collector_source,mod.__file__,'exec'),mod.__dict__)
        def write(name,obj):
            p=root/name;p.parent.mkdir(exist_ok=True,parents=True)
            p.write_text(json.dumps(obj,sort_keys=True));return p
        write('analysis/MODEL_TOKENIZER_FRESH_SHA.json',{'CPU_FIXTURE_ONLY':True})
        write('operations/HUMAN_TASK_AUTHORIZATION.json',{'CPU_FIXTURE_ONLY':True})
        (root/'calibration/collect_fp_once.py').write_text(collector_source)
        (root/'calibration/calibration_math.py').write_text('# CPU_FIXTURE_ONLY\n')
        docs=[]
        for i in range(1,6):
            p=write('calibration/inputs/doc_%02d.json'%i,[i]*256)
            docs.append(dict(input_order=i,role='CAL' if i<5 else 'VAL',token_range=[0,256],State_anchors=[31,95,159,223],document_id='CPU_FIXTURE_ONLY_'+str(i),raw_document_sha256=str(i)*64,new_token_file=str(p.relative_to(root)),new_token_file_sha256=mod.sha(p),input_ids_int64_le_sha256=hashlib.sha256(b''.join(i.to_bytes(8,'little',signed=True) for _ in range(256))).hexdigest()))
        plan=dict(task_id=mod.TASK,AIME_used=False,new_collection_proposal=dict(documents=docs,total_tokens=1280,CAL_next_step_transitions_per_layer_head=1020))
        write('CALIBRATION_CANDIDATE_PLAN.json',plan)
        frozen=['calibration/collect_fp_once.py','calibration/calibration_math.py','CALIBRATION_CANDIDATE_PLAN.json']
        write('CALIBRATION_SOURCE_FREEZE.json',{'files':[dict(path=s,bytes=(root/s).stat().st_size,sha256=mod.sha(root/s)) for s in frozen]})
        control=types.SimpleNamespace(UUIDS=('CPU_FIXTURE_UUID_0','CPU_FIXTURE_UUID_1','CPU_FIXTURE_UUID_2','CPU_FIXTURE_UUID_3'))
        permit=dict(task_id=mod.TASK,scope='FP_COLLECTION_ONLY',ALLOW_GPU='YES',physical_gpu=1,gpu_uuid=control.UUIDS[1],raw_output_root=mod.RAW_ROOT,candidate_plan_sha256=mod.sha(root/'CALIBRATION_CANDIDATE_PLAN.json'),source_freeze_sha256=mod.sha(root/'CALIBRATION_SOURCE_FREEZE.json'),collector_source_sha256=mod.sha(root/'calibration/collect_fp_once.py'),model_manifest_sha256=mod.sha(root/'analysis/MODEL_TOKENIZER_FRESH_SHA.json'),human_authorization_sha256=mod.sha(root/'operations/HUMAN_TASK_AUTHORIZATION.json'))
        write('CALIBRATION_PERMIT.json',permit)
        old=os.environ.get('CUDA_VISIBLE_DEVICES');os.environ['CUDA_VISIBLE_DEVICES']=control.UUIDS[1]
        def reject(label,fn):
            try:fn()
            except (RuntimeError,KeyError,IndexError,FileNotFoundError):checks.append(label);return
            raise AssertionError('accepted illegal fixture:'+label)
        try:
            mod.validate_calibration_permit(root,1,control);checks.append('strict_actual_entry_accepts_complete_CPU_mocked_inputs_no_driver')
            for key,value in [('scope','FORMAL'),('ALLOW_GPU','NO'),('gpu_uuid',control.UUIDS[2]),('physical_gpu',2),('source_freeze_sha256','f'*64),('collector_source_sha256','e'*64),('candidate_plan_sha256','d'*64),('model_manifest_sha256','c'*64),('human_authorization_sha256','b'*64),('raw_output_root','/tmp/foreign')]:
                write('CALIBRATION_PERMIT.json',dict(permit,**{key:value}))
                reject('reject_permit_'+key,lambda:mod.validate_calibration_permit(root,1,control))
            write('CALIBRATION_PERMIT.json',permit)
            reject('reject_GPU0_even_if_free',lambda:mod.validate_calibration_permit(root,0,control))
            reject('reject_wrong_execution_root',lambda:mod.validate_calibration_permit(root.parent,1,control))
            os.environ['CUDA_VISIBLE_DEVICES']=control.UUIDS[2]
            reject('reject_wrong_CVD_UUID',lambda:mod.validate_calibration_permit(root,1,control))
            os.environ['CUDA_VISIBLE_DEVICES']=control.UUIDS[1]
            p=root/docs[0]['new_token_file'];p.write_text('[999]')
            reject('reject_input_bytes_changed',lambda:mod.validate_calibration_permit(root,1,control))
            (root/'calibration/calibration_math.py').write_text('# changed CPU fixture\n')
            reject('reject_frozen_calibration_source_changed',lambda:mod.validate_calibration_permit(root,1,control))
        finally:
            if old is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
            else:os.environ['CUDA_VISIBLE_DEVICES']=old
    return dict(status='PASS_CPU_CALIBRATION_ENTRY_GUARDS',checks=checks,test_count=len(checks),actual_driver_calls=0,model_loads=0,GPU_calls=0,fixtures='EXPLICIT_TEMPORARY_CPU_ONLY_NEVER_GPU_EVIDENCE')
