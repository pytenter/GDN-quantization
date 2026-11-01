"""Compile the exact sealed sampler functions, without importing its old runner."""
import ast,hashlib,json,math,os,random,sys
from pathlib import Path
from types import SimpleNamespace
SAMPLER_FUNCTIONS={'setup_seed','render_input','build_warpers','sample_next','stop_ids'}
CONSTANTS={'TEMPERATURE','TOP_P','TOP_K','MIN_P','PRESENCE_PENALTY','REPETITION_PENALTY','MAX_NEW_TOKENS'}
def extract(path,names,namespace):
    source=Path(path).read_text();nodes=[];found=set();rows=[]
    for n in ast.parse(source).body:
        assigned={t.id for t in n.targets if isinstance(t,ast.Name)} if isinstance(n,ast.Assign) else set()
        selected=({getattr(n,'name',None)}|assigned)&names
        if selected:
            nodes.append(n);found|=selected;rows.append(dict(names=sorted(selected),sha256=hashlib.sha256(ast.get_source_segment(source,n).encode()).hexdigest()))
    if found!=names:raise RuntimeError('Missing sealed protocol AST: '+str(names-found))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return rows

def sampler(root,torch):
    ns=dict(torch=torch,hashlib=hashlib,json=json,math=math,os=os,random=random,sys=sys,Path=Path)
    rows=extract(Path(root)/'protocol/common.py',{'raw_messages','ids_sha256'},ns)
    rows+=extract(Path(root)/'protocol/sampler.py',SAMPLER_FUNCTIONS|CONSTANTS,ns)
    return SimpleNamespace(**{k:v for k,v in ns.items() if k!='__builtins__'}),rows
