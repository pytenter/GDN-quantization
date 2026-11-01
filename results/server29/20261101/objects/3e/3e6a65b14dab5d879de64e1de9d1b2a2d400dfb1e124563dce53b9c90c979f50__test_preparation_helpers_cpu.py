#!/usr/bin/env python3
"""Pure CPU catalog/deployment-receiver mechanics, never a real admission.

All roots, prior versions, negative links and fabricated probe metadata are
outside the scientific root in a TemporaryDirectory. No SSH/tmux/model/Torch
or scientific freeze/launcher is run. Actual receiver code runs in CPU children.
"""
import argparse
import ast
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

ROOT=Path(__file__).resolve().parents[1]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    d=load('_actual_preparation_deploy',ROOT/'operations/deploy_final_pre_gpu_sources.py')
    f=load('_actual_preparation_freeze',ROOT/'operations/freeze_final_qualification_admission_once.py')
    checks=[]
    def check(name,ok):
        if not ok:raise AssertionError(name)
        checks.append(name)
    def rejects(name,fn):
        try:fn()
        except(RuntimeError,AssertionError,ValueError,FileNotFoundError):checks.append(name)
        else:raise AssertionError('Expected rejection: '+name)
    def put(path,v):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(v,sort_keys=True)+'\n')
    def file_row(root,p):
        b=p.read_bytes();return dict(path=str(p.relative_to(root)),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
    with tempfile.TemporaryDirectory(prefix='resident_prepare_helpers_CPU_')as temporary:
        top=Path(temporary);root=top/'synthetic_science';root.mkdir()
        put(root/'ROOT.json',{'CPU_SYNTHETIC_ROOT':True})
        grant=dict(prior_source_freeze_sha256=d.PRIOR153,execution_attempt=root.name,qualification_authorized=True,formal_conditionally_authorized=True)
        put(root/'HUMAN_REPAIR_AUTHORIZATION.json',grant)
        put(root/'operations/P1_AUTHORIZED_STOP_COMPLETION/USER_STOP_RECEIPT.json',{'CPU_SYNTHETIC_NOT_REAL_P1':True})
        raw=[dict(path='calibration/test_%02d.npz'%i,bytes=10,sha256='0'*64)for i in range(25)]
        put(root/'operations/REMOTE_ONLY25_ASSET_COPY.json',dict(status='PASS_REMOTE_ONLY_ORIGINAL25_ASSET_INDEPENDENT_COPY',files=raw))
        probes=[]
        for i in(1,2):
            p=root/('operations/fresh_%d.json'%i);put(p,{'CPU_SYNTHETIC_NATIVE_RECEIPT':i});probes.append(file_row(root,p))
        closure=dict(cpu_probe_receipts=probes,files=[])
        put(root/'analysis/RUNTIME_NATIVE_CLOSURE.json',closure)
        put(root/'analysis/other_input.json',{'input':1})
        (root/'operations/helper.py').write_text('CPU_ONLY_HELPER=True\n')
        selected=d.paths(root);names={str(p.relative_to(root))for p in selected}
        check('catalog_includes_two_bound_CPU_receipts',all(x['path']in names for x in probes))
        check('catalog_includes_native_manifest_and_tools','analysis/RUNTIME_NATIVE_CLOSURE.json'in names and'operations/helper.py'in names)
        check('catalog_keeps_dynamic_source_count',len(selected)==len(names))
        zero=d.probe_counts(root)
        check('no_probe_does_not_invent_GPU_work',zero['engineering_native_loader_GPU_probes']==zero['total_GPU_work_dispatches']==0)
        rejects('scientific_freeze_plan_requires_actual_completed_probe',lambda:f.source_payload(root))
        started=root/'operations/loader_native_probe/STARTED.json';put(started,{'CPU_SYNTHETIC':True})
        rejects('active_probe_blocks_source_replacement',lambda:d.paths(root));started.unlink()
        failed=dict(status='FAIL_ENGINEERING_LOAD_ONLY_PROBE',actual_wait_returncode=-11,all_exact_own_processes_exited=True,own_GPU_UUID_released=True,
                    model_forward_calls=0,QDQ_calls=0,AIME_generations=0,counts_as_scientific_qualification=False)
        put(root/d.PROBE_COMPLETION,failed)
        rejects('real_nonzero_exit_cannot_be_relabelled_PASS',lambda:d.probe_counts(root))
        fake_pass=dict(failed,status='PASS_ENGINEERING_LOAD_ONLY_PROBE_EXITED_RELEASED')
        put(root/d.PROBE_COMPLETION,fake_pass)
        rejects('PASS_string_does_not_override_wait_minus11',lambda:d.probe_counts(root))
        fake_pass['actual_wait_returncode']=0;put(root/d.PROBE_COMPLETION,fake_pass)
        counts=d.probe_counts(root)
        check('engineering_work1_and_scientific_work0_counted_separately',counts==dict(scientific_qualification_GPU_dispatches=0,engineering_native_loader_GPU_probes=1,total_GPU_work_dispatches=1))
        payload=f.source_payload(root);freeze=payload['freeze']
        check('freeze_source_metadata_honestly_counts_prior_engineering_probe',freeze['GPU_dispatches_at_freeze']==1 and freeze['scientific_qualification_GPU_dispatches_at_freeze']==0 and freeze['engineering_native_loader_GPU_probes_at_freeze']==1)
        check('freeze_plan_is_not_GPU_qualification',freeze['CPU_PASS_is_GPU_PASS']is False and freeze['loader_probe_is_scientific_qualification']is False)
        check('freeze_plan_does_not_write_source_or_permit',not(root/'SOURCE_FREEZE.json').exists()and not(root/'GPU_PERMIT.json').exists())
        saved=(root/probes[0]['path']).read_bytes();(root/probes[0]['path']).write_bytes(saved+b' ')
        rejects('changed_bound_CPU_receipt_rejected',lambda:d.paths(root));(root/probes[0]['path']).write_bytes(saved)
        invalid=copy.deepcopy(closure);invalid['cpu_probe_receipts'][0]['path']='../outside.json';put(root/'analysis/RUNTIME_NATIVE_CLOSURE.json',invalid)
        rejects('source_bound_parent_escape_rejected',lambda:d.paths(root));put(root/'analysis/RUNTIME_NATIVE_CLOSURE.json',closure)
        target=root/'analysis/negative_link';target.symlink_to(top/'missing')
        rejects('all_tree_dangling_symlink_rejected_not_silently_skipped',lambda:d.paths(root));target.unlink()
        outside=top/'outside';outside.mkdir();(outside/'KEEP').write_bytes(b'KEEP');target.symlink_to(outside,target_is_directory=True)
        rejects('all_tree_directory_symlink_rejected',lambda:d.paths(root));target.unlink()
        check('negative_symlink_destination_never_touched',(outside/'KEEP').read_bytes()==b'KEEP')
        bad=dict(grant,prior_source_freeze_sha256='c8741585a335e9cd66c68aa7154126b64ddc8e7461b7a67cfeb00e996c5d077a');put(root/'HUMAN_REPAIR_AUTHORIZATION.json',bad)
        rejects('obsolete134_authorization_not_accepted_as153',lambda:d.paths(root));put(root/'HUMAN_REPAIR_AUTHORIZATION.json',grant)
        receiver=top/'receiver';receiver.mkdir()
        def receive(files):
            b=io.BytesIO()
            with tarfile.open(fileobj=b,mode='w')as t:
                for name,value in files.items():
                    entry=tarfile.TarInfo(name);entry.size=len(value);t.addfile(entry,io.BytesIO(value))
            return subprocess.run([sys.executable,'-B','-c',d.RECEIVER,str(receiver)],input=b.getvalue(),capture_output=True,
                                  env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'))
        auth=json.dumps(grant).encode();r=receive({'HUMAN_REPAIR_AUTHORIZATION.json':auth,'src/test.py':b'FIRST'})
        check('actual_receiver_CPU_child_first_regular_bytes_pass',r.returncode==0 and(receiver/'src/test.py').read_bytes()==b'FIRST')
        r=receive({'src/test.py':b'SECOND'})
        oldsha=hashlib.sha256(b'FIRST').hexdigest()
        check('actual_receiver_changed_bytes_preserves_prior_exact_version',r.returncode==0 and(receiver/'operations/PRE_FREEZE_DEPLOYMENT_VERSIONS'/oldsha/'src/test.py').read_bytes()==b'FIRST')
        (receiver/'SOURCE_FREEZE.json').write_bytes(b'CPU_SYNTHETIC')
        check('actual_receiver_rejects_post_freeze_write',receive({'src/test.py':b'BAD'}).returncode!=0 and(receiver/'src/test.py').read_bytes()==b'SECOND');(receiver/'SOURCE_FREEZE.json').unlink()
        (receiver/'src/link.py').symlink_to(outside/'KEEP')
        check('actual_receiver_refuses_symlink_destination',receive({'src/link.py':b'BAD'}).returncode!=0 and(outside/'KEEP').read_bytes()==b'KEEP');(receiver/'src/link.py').unlink()
        put(receiver/'operations/loader_native_probe/STARTED.json',{'CPU_SYNTHETIC':True})
        check('actual_receiver_refuses_active_engineering_probe',receive({'src/test.py':b'BAD'}).returncode!=0)
        put(receiver/d.PROBE_COMPLETION,failed)
        check('actual_receiver_refuses_failed_engineering_probe',receive({'src/test.py':b'BAD'}).returncode!=0)
        check('receiver_rejections_leave_old_regular_bytes_intact',(receiver/'src/test.py').read_bytes()==b'SECOND')
        for name,embedded in [('deploy',d.RECEIVER),('freeze',f.PROGRAM)]:
            compile(embedded,'actual_'+name+'_embedded_program','exec');checks.append('embedded_'+name+'_target_program_compiles')
        launch=ast.parse((ROOT/'operations/launch_unique_offline.py').read_text())
        code=next(ast.literal_eval(x.value)for x in launch.body if isinstance(x,ast.Assign)and any(isinstance(n,ast.Name)and n.id=='PROGRAM'for n in x.targets))
        compile(code,'actual_launch_embedded_program','exec');checks.append('embedded_unique_launcher_compiles')
        check('no_Torch_import_and_no_GPU_or_SSH_children','torch'not in sys.modules)
    pins={n:hashlib.sha256((ROOT/'operations'/n).read_bytes()).hexdigest()for n in('deploy_final_pre_gpu_sources.py','freeze_final_qualification_admission_once.py','launch_unique_offline.py','test_preparation_helpers_cpu.py')}
    result=dict(status='PASS_CPU_PREPARATION_CATALOG_AND_ACTUAL_RECEIVER_MECHANICS_ONLY',check_count=len(checks),checks=checks,source_pins=pins,
                Torch_imported=False,real_GPU_dispatches=0,real_SSH_calls=0,real_freeze_or_launcher_executed=False,
                all_synthetic_roots_and_negative_links_outside_science_ROOT=True,
                does_not_qualify_real_failed_loader_probe=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x')as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(status=result['status'],check_count=len(checks))))


if __name__=='__main__':main()
