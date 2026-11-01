"""Read-only bounded extraction from already qualified FP raw; no GPU imports."""
import base64,hashlib,json,shlex,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1]
P=R.parent/'QWEN_GDN_HISTORICAL_RUNTIME_GPU_RECOVERY_V1/gpu_execution_20261019_v1/delivery/payload/execution/NUMERIC_FRESH0'
remote='/data/zypan/analyses/qwen_gdn_historical_runtime_gpu_recovery_v1/gpu_execution_20261019_v1/payload/execution/NUMERIC_FRESH0'
requests=[]
for t in (0,15,31):
    meta=json.loads((P/f'TOKEN_{t:03}.json').read_text())
    rows=[]
    for layer in (0,16,30):
        row=next(x for x in meta['tensors'] if x['name']==f'cache/layers/{layer}/recurrent_states/int:0')
        assert row['shape']==[1,32,128,128] and row['dtype']=='torch.float32'
        rows.append(dict(row,layer=layer))
    requests.append(dict(token=t,path=remote+f'/raw/token_{t:03}.bin',sha256=meta['raw_sha256'],bytes=meta['raw_bytes'],rows=rows))
code=r'''import base64,hashlib,json,sys
from pathlib import Path
req=json.load(sys.stdin);out=[]
for r in req:
 p=Path(r['path']);a=p.stat();raw=p.read_bytes();z=p.stat()
 assert (a.st_ino,a.st_size,a.st_mtime_ns)==(z.st_ino,z.st_size,z.st_mtime_ns)
 assert len(raw)==r['bytes'] and hashlib.sha256(raw).hexdigest()==r['sha256']
 for row in r['rows']:
  allstate=raw[row['offset']:row['offset']+row['bytes']]
  assert hashlib.sha256(allstate).hexdigest()==row['sha256']
  for head in (0,15,31):
   small=allstate[head*65536:(head+1)*65536]
   out.append(dict(token=r['token'],layer=row['layer'],head=head,parent_bin=r['path'],parent_sha256=r['sha256'],parent_state_sha256=row['sha256'],base64=base64.b64encode(small).decode(),sha256=hashlib.sha256(small).hexdigest(),bytes=len(small)))
print(json.dumps(out))'''
command=shlex.join(['/data/ydai/miniconda3/envs/bitdecode/bin/python','-B','-c',code])
p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-S','/tmp/qwen_gdn_replication_20261013.sock','zypan@202.38.247.55',command],input=json.dumps(requests),text=True,capture_output=True,check=True,timeout=90)
dest=R/'fixtures/real_states';dest.mkdir(parents=True,exist_ok=True)
records=[]
for row in json.loads(p.stdout):
    raw=base64.b64decode(row.pop('base64'));assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
    name=f"t{row['token']:03}_l{row['layer']:02}_h{row['head']:02}.f32"
    with (dest/name).open('xb') as f:f.write(raw)
    row['file']='fixtures/real_states/'+name;row['shape']=[1,1,128,128];records.append(row)
manifest=dict(status='PASS_READ_ONLY_PARENT_RAW_EXTRACTION',selection='tokens0,15,31 x layers0,16,30 x heads0,15,31; fixed before codec scoring',new_forward=False,GPU_queries=0,GPU_dispatches=0,bytes=sum(x['bytes'] for x in records),records=records)
(R/'fixtures/REAL_STATE_INPUT_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:v for k,v in manifest.items() if k!='records'}))
