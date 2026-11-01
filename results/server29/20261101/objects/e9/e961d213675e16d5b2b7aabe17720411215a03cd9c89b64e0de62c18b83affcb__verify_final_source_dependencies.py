#!/usr/bin/env python3
"""Post-exit fresh streaming SHA check via unchanged native CPU admission."""
import argparse, hashlib, importlib.util, json, pathlib, sys, time
sys.dont_write_bytecode=True

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=pathlib.Path);p.add_argument('--run',action='store_true');a=p.parse_args();r=a.root.resolve();payload=r/'payload'
    if not a.run:
        print(json.dumps(dict(status='PREVIEW_ONLY_NO_HASH_NO_GPU_NO_WRITES')));return
    proof=json.loads((r/'analysis/POST_GPU_RELEASE_ADMISSION.json').read_text())
    assert proof['status']=='PASS_ALL_OWN_PROCESSES_EXITED_ALL_UUIDS_RELEASED'
    assert all(proof[k]is True for k in ['controller_exit_verified','controller_exited','all_owned_processes_exited','all_owned_GPU_resources_released'])
    out=r/'analysis/FINAL_SOURCE_DEPENDENCY_VERIFICATION.json';assert not out.exists()
    spec=importlib.util.spec_from_file_location('post_exit_native_admission',payload/'scripts/restoration_controller.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    begin=time.monotonic();checked=m.authorize(payload,'YES',payload/'HUMAN_GPU_RESTORE_AUTHORIZATION.json')
    freeze=json.loads((payload/'SOURCE_FREEZE.json').read_text());binding={}
    for name in checked['plan']['binding_files']:
        for row in m.binding_rows(m.read(payload/name)):
            path=str(pathlib.Path(row.get('resolved_path',row['path'])).resolve())
            if path in binding:assert(binding[path]['bytes'],binding[path]['sha256'])==(row['bytes'],row['sha256'])
            binding[path]=dict(path=path,bytes=row['bytes'],sha256=row['sha256'])
    assert len(binding)==checked['external_files']==249 and len(freeze['files'])==407
    receipt=dict(task_id=m.TASK,status='PASS_FRESH_POST_GPU407_AND249_EXTERNAL_STREAM_SHA_BYTES',source_freeze_sha256=checked['source_freeze_sha256'],parent_effective_source_freeze_sha256=checked['parent_effective_source_freeze_sha256'],authorization_sha256=checked['authorization_sha256'],plan_sha256=checked['plan_sha256'],files_verified=407,inherited_source_files=403,amended_source_files=4,external_files_verified=249,model_weight_shards_verified=4,source_files=freeze['files'],external_files=sorted(binding.values(),key=lambda x:x['path']),elapsed_same_host_monotonic_seconds=time.monotonic()-begin,GPU_queries=0,GPU_dispatches=0,Torch_imported='torch'in sys.modules,source_or_model_mutated=False,large_model_downloaded_to_Mac=False,verification_script_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())
    assert receipt['Torch_imported']is False
    with out.open('x')as f:json.dump(receipt,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:v for k,v in receipt.items()if k not in ['source_files','external_files']},ensure_ascii=False))

if __name__=='__main__':main()
