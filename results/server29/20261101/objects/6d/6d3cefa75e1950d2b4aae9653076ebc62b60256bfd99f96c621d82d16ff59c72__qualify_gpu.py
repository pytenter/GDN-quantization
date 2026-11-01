import argparse,json,os,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("--root",type=Path);p.add_argument("--unit");p.add_argument("--gpu",type=int);a=p.parse_args()
sys.path.insert(0,str(a.root/"src"));import resident_execution_control as C
time.sleep(.25)
u=next(x for x in C.read(a.root/"QUALIFICATION_JOB_MANIFEST.json")["units"]if x["id"]==a.unit)
t=a.root/"qualification"/a.unit;t.mkdir(parents=True)
C.atomic_new(t/"STARTED.json",dict(identity=C.process_identity(os.getpid()),synthetic=True))
C.atomic_new(t/"PROGRESS.json",dict(phase="SYNTHETIC_CPU_ONLY",token=1))
if a.unit=="GPU2_FAIL":
 time.sleep(.5);C.atomic_new(t/"FIRST_FAILURE.json",dict(status="SYNTHETIC_EXPECTED_FAILURE"));raise SystemExit(7)
if "HOLD"in a.unit:time.sleep(45)
C.atomic_new(t/"RESULT.json",dict(status="PASS",unit=u,result=dict(synthetic=True)))
