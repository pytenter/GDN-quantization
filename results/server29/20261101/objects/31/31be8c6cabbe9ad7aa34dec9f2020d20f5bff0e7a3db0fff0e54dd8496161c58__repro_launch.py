"""Set the frozen scientific environment before interpreter startup."""
from pathlib import Path
import argparse,json,os
R=Path('/data/zypan/worktrees/ling-long-horizon-observability-bridge-v3')
Q=R/'provenance_capsule_v3/scientific_reproducibility_v3_virtual_namespace'
PY='/data/zypan/envs/ling-kda/bin/python'
def main():
    p=argparse.ArgumentParser();p.add_argument('mode');p.add_argument('run_id');a=p.parse_args()
    profile=json.loads((Q/'launcher_profile.json').read_text());env=dict(os.environ)
    env.update(profile['set'])
    for name in profile['unset']:env.pop(name,None)
    env['PYTHONDONTWRITEBYTECODE']='1'
    os.chdir(R)
    os.execve(PY,[PY,'-B',str(Q/'repro_runner.py'),a.mode,a.run_id],env)
if __name__=='__main__':main()
