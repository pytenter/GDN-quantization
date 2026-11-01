"""Three meaningful independent CPU scientific/integration audits per fresh suite."""
import os,sys,subprocess,json,time,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1]
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
rep=int(sys.argv[1]);out=R/'qualification'/('CPU_PASS_'+str(rep));out.mkdir(exist_ok=False)
programs=[
 ('codec',"import sys, json;from pathlib import Path;sys.path.insert(0,"+repr(str(R/'src'))+");import torch;torch.set_num_threads(1);torch.set_num_interop_threads(1);from codec_qualification import qualify;r=qualify("+repr(str(out/'codec'))+",'cpu');assert not torch.cuda.is_initialized();print('RESULT_JSON='+json.dumps(r,sort_keys=True))"),
 ('tiny_math',str(R/'scripts/cpu_qualification.py')),
 ('integration',str(R/'scripts/test_execution_cpu.py'))]
reports={}
for name,p in programs:
    cmd=[sys.executable,'-u','-B','-c',p] if name=='codec' else [sys.executable,'-u','-B',p]
    started=time.monotonic();v=subprocess.run(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,timeout=600)
    (out/(name+'.log')).write_bytes(v.stdout+v.stderr)
    if v.returncode:
        (out/'FAILURE.json').write_text(json.dumps({'stage':name,'returncode':v.returncode,'GPU_dispatches':0,'error_log':name+'.log'},indent=2)+'\n')
        sys.stdout.buffer.write(v.stdout+v.stderr);raise SystemExit(v.returncode)
    lines=[x for x in v.stdout.decode().splitlines() if x.startswith('RESULT_JSON=')]
    assert len(lines)==1
    result=json.loads(lines[0].split('=',1)[1]);(out/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n');reports[name]=result
    print(name,'PASS','seconds',time.monotonic()-started,flush=True)
summary={'status':'PASS_ALL_CPU_CODEC_TINY_MATH_INTEGRATION','reports':reports,'CUDA_VISIBLE_DEVICES':'','GPU_dispatches':0}
(out/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');print('CPU_SUITE_PASS',rep,flush=True)
