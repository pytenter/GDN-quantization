"""Exact parent scientific modules; redirect operational namespaces only."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import importlib.util
from transport import GPUS, UUIDS, DEST, PD, mapped, remap_value

TASK = 'LING_LONG_HORIZON_PROPAGATION_CONFIRM_V1'
D = Path(__file__).resolve().parents[1]
R = D.parents[1]
PARENT_ROOT = Path('/data01/user2/worktrees/ling-propagation-surrogate-validation-v1')
PARENT = PARENT_ROOT / 'experiments/LING_PROPAGATION_SURROGATE_VALIDATION_V1'
PARENT_COMMIT = 'dc602d2d2d3b78346b6e89d9b6a71a9ad9ddc53b'
REMOTE = str(R).startswith('/data/zypan/')
SSH = ['ssh','-i','/data01/user2/.ssh/id_ed25519_ling_remote','-o','BatchMode=yes','-o','ServerAliveInterval=15','zypan@202.38.247.29']
if REMOTE:
    from transport import scientific_modules
    c,capture,core,parent_worker,stats=scientific_modules()
    import transition
    PARENT_ROOT=mapped(PARENT_ROOT);PARENT=mapped(PARENT)
else:
    sys.path.insert(0, str(PARENT / 'scripts'))
    spec = importlib.util.spec_from_file_location('common', PARENT / 'scripts/common.py')
    c = importlib.util.module_from_spec(spec);sys.modules['common']=c;spec.loader.exec_module(c)
    c.D,c.R,c.TASK=D,R,TASK
    import replay_core as core
    import transition,capture
    import worker as parent_worker
    import surrogate_statistics as stats
    import subprocess,json
    def hardware():
        query='nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,driver_version --format=csv,noheader,nounits'
        lines=subprocess.check_output(SSH+[query],text=True).splitlines()
        rows=[]
        for line in lines:
            a=[v.strip() for v in line.split(',')];rows.append(dict(physical_gpu=int(a[0]),gpu_uuid=a[1],name=a[2],total_mib=int(a[3]),used_mib=int(a[4]),driver=a[5]))
        index={x['physical_gpu']:x for x in rows}
        for g in GPUS:assert index[g]['gpu_uuid']==UUIDS[g] and '3090' in index[g]['name']
        return dict(host='nlpg-SYS-4029GP-TRT',gpus=rows)
    c.hardware=hardware;c.UUIDS=UUIDS
for module in (c,core,transition,capture,parent_worker,stats):
    assert Path(module.__file__).resolve().parent==PARENT/'scripts'
