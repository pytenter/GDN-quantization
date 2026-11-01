from pathlib import Path
import sys,hashlib,json
sys.path.insert(0,str(Path(__file__).resolve().parent));import local_ops as o
names=sys.argv[1:]
rows=[]
for n in names:
 code="from pathlib import Path\nimport sys\nsys.stdout.buffer.write((Path("+repr(o.REMOTE)+")/"+repr(n)+").read_bytes())"
 raw=o.call(code);assert len(raw)<5000000
 digest=hashlib.sha256(raw).hexdigest()
 proof=json.loads(o.call("from pathlib import Path\nimport hashlib,json\np=Path("+repr(o.REMOTE+'/'+n)+");raw=p.read_bytes();print(json.dumps(dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())))"))
 assert proof==dict(bytes=len(raw),sha256=digest)
 p=o.BASE/n;p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_bytes()==raw
 else:p.write_bytes(raw)
 rows.append(dict(path=n,**proof))
(o.BASE/'RESULT_DOWNLOAD_PROOF.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows))
