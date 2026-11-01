from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parent));import local_ops as o
c="""from pathlib import Path
import json
r=Path(__ROOT__);l=json.loads((r/'GPU_RESOURCE_LEDGER.json').read_bytes());out={'status':l['status'],'GPUh':l['cumulative_GPU_hours'],'live':l.get('live_unit'),'results':[]}
for p in sorted((r/'execution').glob('*/RESULT.json')):
 j=json.loads(p.read_bytes());out['results'].append({'unit':p.parent.name,'TPS':j['tokens_per_second'],'profile':j['profile']})
for n in ['FIRST_FAILURE.json','execution/BATCH_RESULT.json']:
 p=r/n;out[n]=json.loads(p.read_bytes()) if p.exists() else None
print(json.dumps(out))""".replace('__ROOT__',repr(o.REMOTE))
raw=o.call(c);(o.BASE/'LIVE_COMPACT.json').write_bytes(raw);print(raw.decode())
