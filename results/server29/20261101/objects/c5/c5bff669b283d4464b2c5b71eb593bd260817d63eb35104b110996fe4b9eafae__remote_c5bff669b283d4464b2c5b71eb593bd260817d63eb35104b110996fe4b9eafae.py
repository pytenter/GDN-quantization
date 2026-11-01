import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,json
names=['tokenizer.json','tokenizer_config.json','merges.txt','vocab.json','chat_template.jinja','config.json','configuration.json','model.safetensors.index.json','preprocessor_config.json','video_preprocessor_config.json']
out=[]
for n in names:
 p=Path('/data/zypan/modelscope_models/Qwen3.5-9B')/n;a=p.stat();h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 z=p.stat();stable=(a.st_ino,a.st_size,a.st_mtime_ns)==(z.st_ino,z.st_size,z.st_mtime_ns)
 out.append(dict(path=str(p),bytes=z.st_size,sha256=h.hexdigest(),mtime_ns=z.st_mtime_ns,inode=z.st_ino,status='STABLE_CURRENT_CPU_HASH' if stable else 'UNSTABLE',historical_full_tokenizer_identity='UNKNOWN'))
print(json.dumps(out))