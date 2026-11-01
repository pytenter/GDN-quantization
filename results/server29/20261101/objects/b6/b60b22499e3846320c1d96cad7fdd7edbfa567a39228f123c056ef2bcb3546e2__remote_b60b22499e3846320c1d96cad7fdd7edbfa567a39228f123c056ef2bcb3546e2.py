import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.dont_write_bytecode=True
from pathlib import Path
import json
p=Path('/data/zypan/modelscope_models/Qwen3.5-9B')
print(json.dumps([dict(path=str(x),bytes=x.stat().st_size,symlink=x.is_symlink()) for x in sorted(p.iterdir()) if x.is_file()]))