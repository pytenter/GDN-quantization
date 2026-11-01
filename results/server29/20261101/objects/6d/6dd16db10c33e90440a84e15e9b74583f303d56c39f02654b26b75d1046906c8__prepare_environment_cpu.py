"""Independent read-only target inventory; no CUDA/model load or remote writes."""
import datetime
import importlib.util
import json
from pathlib import Path

L = Path(__file__).resolve().parents[1]
W = L.parent / 'QWEN_GDN_SYM8_G32_WSCALE_ROWBAL_AIME26_SEED1_V1_BYTEVIEW_REPAIR_20261025'
G = L.parent / 'QWEN_GDN_GROUP64_FOUR_CODEC_AIME26_SEED1_V1'
spec = importlib.util.spec_from_file_location('transport', G / 'operations/remote.py')
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
models = json.loads((W / 'analysis/MODEL_TOKENIZER_FRESH_SHA.json').read_text())['files']
old_native = json.loads((G / 'revisions/three_gpu_reuse57_resume_v7/node_profiles/server55_gpu0/analysis/EXTERNAL_NATIVE_BINDINGS.json').read_text())['files']
expected = {x['path']: x for x in old_native
            if '/qwen-gdn-group64-historical-user-site-v1/' not in x['path']}
expected.update({x['path']: x for x in models})
code = '''
import os,json,hashlib,pathlib,datetime,subprocess
expected=EXPECTED
def row(path):
 p=pathlib.Path(path);s=p.stat();h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 t=p.stat();assert (s.st_size,s.st_ino,s.st_mtime_ns)==(t.st_size,t.st_ino,t.st_mtime_ns)
 return dict(path=str(p),bytes=s.st_size,sha256=h.hexdigest(),inode=s.st_ino,mtime_ns=s.st_mtime_ns,device=s.st_dev)
native=[];models=[];python=[]
for path,x in sorted(expected.items()):
 r=row(path);assert (r['bytes'],r['sha256'])==(x['bytes'],x['sha256']),path
 (models if '/modelscope_models/' in path else native).append(r)
roots=['/data/ydai/miniconda3/envs/bitdecode/lib/python3.10/site-packages',
 '/data/zypan/.local/lib/python3.10/site-packages',
 '/data/zypan/transformers-qwen35/src',
 '/data/zypan/experiments/qwen35_gdn_quant']
seen={r['path'] for r in native+models}
for root in roots:
 assert pathlib.Path(root).is_dir(),root
 for d,dirs,files in os.walk(root):
  dirs[:]=sorted(v for v in dirs if v not in ('__pycache__','.git','tests','test'))
  for name in sorted(files):
   p=pathlib.Path(d)/name
   if p.is_file() and (name.endswith('.py') or '.so' in name or name in ('METADATA','entry_points.txt')) and str(p) not in seen:
    r=row(p);python.append(r);seen.add(str(p))
out=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),GPU_calls=0,model_loads=0,remote_writes=0,
 model_files=models,native_files=native,python_files=python,roots=roots,
 nvml=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.free','--format=csv,noheader,nounits'],text=True),
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True))
print(json.dumps(out))
'''.replace('EXPECTED', repr(expected))
out = transport.run(code, role='55', python='/data/ydai/miniconda3/envs/bitdecode/bin/python', timeout=900)
(L / 'analysis').mkdir(exist_ok=True)
for name, field in [('MODEL_TOKENIZER_FRESH_SHA.json', 'model_files'),
                    ('EXTERNAL_NATIVE_BINDINGS.json', 'native_files'),
                    ('EXTERNAL_PYTHON_BINDINGS.json', 'python_files')]:
    value = dict(status='PASS_FRESH_TARGET_CPU_SHA_BYTES', UTC=out['UTC'],
                 GPU_calls=0, model_loads=0, files=out[field], count=len(out[field]))
    with (L / 'analysis' / name).open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2); f.write('\n')
summary = {k: v for k, v in out.items() if not k.endswith('_files')}
summary.update(status='PASS_TARGET_CPU_FULL_MODEL17_NATIVE_PYTHON_FRESH_SHA',
               model_count=len(out['model_files']), native_count=len(out['native_files']),
               python_count=len(out['python_files']))
with (L / 'operations/CPU_TARGET_ENVIRONMENT_ADMISSION.json').open('x') as f:
    json.dump(summary, f, sort_keys=True, indent=2); f.write('\n')
print(json.dumps(summary))
