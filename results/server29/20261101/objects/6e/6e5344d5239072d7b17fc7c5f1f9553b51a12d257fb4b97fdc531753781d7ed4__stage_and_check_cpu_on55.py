#!/usr/bin/env python3
"""Install isolated frozen small-source preparation and perform CPU-only checks."""
import gzip,hashlib,io,json,os,shlex,subprocess,tarfile
from pathlib import Path
R=Path(__file__).resolve().parents[1]
REMOTE=r'''
import os,sys,hashlib,json,pathlib,tarfile,io,subprocess
root=pathlib.Path(sys.argv[1]);expected=sys.argv[2];raw=sys.stdin.buffer.read();assert hashlib.sha256(raw).hexdigest()==expected
root.mkdir(parents=True,exist_ok=False)
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as t:
 for member in t:
  p=pathlib.PurePosixPath(member.name);assert not p.is_absolute()and'..'not in p.parts and member.isfile()
  q=root/member.name;q.parent.mkdir(parents=True,exist_ok=True)
  with q.open('xb')as f:f.write(t.extractfile(member).read())
  q.chmod(member.mode&0o777)
assert json.loads((root/'ROOT.json').read_text())['ALLOW_GPU_RESTORE']=='NO'
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
p=subprocess.run([sys.executable,'-B',str(root/'scripts/candidate_check.py'),'--check'],env=env,capture_output=True,text=True,timeout=240)
assert p.returncode==0,p.stderr
cpu="""
import os,sys,pathlib,hashlib,json,importlib.util
sys.dont_write_bytecode=True
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
import torch
def denied(*a,**kw):raise RuntimeError('GPU initialization forbidden')
torch.cuda._lazy_init=denied
r=pathlib.Path(sys.argv[1]);f=r/'candidate/source_tree/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/qwen_gdn/aime26/81920/run_qwen_aime26_formal.py'
sys.path.insert(0,str(f.parent));s=importlib.util.spec_from_file_location('isolated_old_import_cpu_only',f);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
import aime26_common as common
assert hashlib.sha256(f.read_bytes()).hexdigest()=='9ff56d00b6b2dbaf9be3c3186bd30be8d3b9cc9d9b580a680316bcc8c01247dd'
assert hashlib.sha256(pathlib.Path(common.__file__).read_bytes()).hexdigest()=='34a320d20a48cda8d0a2a194914f3fa04f8c32bf8c6c533f5f9916fc5d6f9df7'
assert callable(m.load_model_and_tokenizer) and len(m.GDN_LAYERS)==24
print(json.dumps(dict(status='PASS_ISOLATED_ORIGINAL_LOADER_CPU_IMPORT_ONLY',model_loaded=False,GPU_queries=0,GPU_dispatches=0,GPU_initialization_guard='DENIED',runner_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),common_sha256=hashlib.sha256(pathlib.Path(common.__file__).read_bytes()).hexdigest())))
"""
q=subprocess.run([sys.executable,'-B','-c',cpu,str(root)],env=env,capture_output=True,text=True,timeout=120)
assert q.returncode==0,q.stderr
print(json.dumps(dict(status='PASS_ISOLATED_CPU_STAGE_AND_CURRENT_RUNTIME_IDENTITY',root=str(root),bundle_sha256=expected,candidate_check=json.loads(p.stdout),isolated_loader_CPU_check=json.loads(q.stdout),GPU_queries=0,GPU_dispatches=0,model_loaded=False)))
'''
def main():
    root=json.loads((R/'ROOT.json').read_text());assert root['ALLOW_GPU_RESTORE']=='NO'
    freeze=json.loads((R/'SOURCE_FREEZE.json').read_text());names=[x['path']for x in freeze['files']]+['SOURCE_FREEZE.json','ROOT.json','HUMAN_GPU_RESTORE_AUTHORIZATION.json']
    output=io.BytesIO()
    with gzip.GzipFile(fileobj=output,mode='wb',mtime=0)as gz,tarfile.open(fileobj=gz,mode='w')as tf:
        for name in names:
            p=R/name;data=p.read_bytes();entry=tarfile.TarInfo(name);entry.size=len(data);entry.mode=p.stat().st_mode&0o777;entry.mtime=0;tf.addfile(entry,io.BytesIO(data))
    data=output.getvalue();h=hashlib.sha256(data).hexdigest();bundle=R/'operations/CPU_CANDIDATE_STAGE.tar.gz'
    with bundle.open('xb')as f:f.write(data)
    command=shlex.join(['/data/ydai/miniconda3/envs/bitdecode/bin/python','-B','-c',REMOTE,root['remote_root'],h])
    proc=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-S',root['ssh_control_path'],root['remote_user']+'@'+root['remote_host'],command],input=data,capture_output=True,timeout=420)
    (R/'operations/CPU_REMOTE_STAGE_STDOUT.log').write_bytes(proc.stdout);(R/'operations/CPU_REMOTE_STAGE_STDERR.log').write_bytes(proc.stderr)
    assert proc.returncode==0,proc.stderr.decode(errors='replace')
    result=json.loads(proc.stdout);(R/'qualification/REMOTE_ISOLATED_CPU_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
