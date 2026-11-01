"""Read-only stdlib local mirror seal verification; imports no Torch runtime."""
import argparse,hashlib,importlib.util,json,pathlib,time

def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def main(a):
    module=pathlib.Path(__file__).with_name('qualify_structure.py')
    spec=importlib.util.spec_from_file_location('structure_definition',module)
    q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
    parent=pathlib.Path(a.parent);rows=[]
    for label,(h,count) in q.EXPECTED_SEALS.items():
        root=parent/label;manifest=root/'hashes/artifact_sha256.txt'
        assert sha(manifest)==h
        lines=manifest.read_text().splitlines();assert len(lines)==count
        for line in lines:
            expected,rel=line.split('  ',1);assert sha(root/rel)==expected,(label,rel)
        rows.append(dict(root=str(root),manifest_sha256=h,verified_files=count,status='PASS_UNCHANGED',
                         receipt_sha256=sha(root/'hashes/DELIVERY_VERIFICATION.json')))
    remote=json.loads(pathlib.Path(a.remote_receipt).read_text())
    assert remote['status']=='PASS' and remote['cuda_initialized'] is False
    assert remote['script_sha256']==sha(module)
    assert remote['formal_source_clue']['sha256']==sha(module.with_name('source_clue_run_qwen_aime26_formal.py'))
    result=dict(status='PASS',scope='Local sealed mirror files and retrieved remote CPU receipt only',seals=rows,
                model_loaded=False,GPU_HOURS_USED=0,torch_imported=False,script_sha256=sha(__file__),
                remote_CPU_receipt_sha256=sha(a.remote_receipt),completed_unix=time.time())
    with pathlib.Path(a.output).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(status='PASS',sealed_files=sum(x['verified_files'] for x in rows),output=a.output)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--remote-receipt',required=True);p.add_argument('--output',required=True);main(p.parse_args())
