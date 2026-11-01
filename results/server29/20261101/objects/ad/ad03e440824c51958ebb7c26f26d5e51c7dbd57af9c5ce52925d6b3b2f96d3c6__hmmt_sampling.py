"""Original sealed Qwen sampler functions, only requested temperature/cap changed."""
import ast,hashlib,json,math,os,random,sys
from pathlib import Path
from types import SimpleNamespace
def sampler(root,torch):
    root=Path(root)
    ns=dict(torch=torch,hashlib=hashlib,json=json,math=math,os=os,random=random,sys=sys,Path=Path)
    # Reuse exact historical AST rather than importing the old AIME runner.
    for filename,names in [('common.py',{'ids_sha256'}),
                          ('sampler.py',{'setup_seed','build_warpers','sample_next','stop_ids','TEMPERATURE','TOP_P','TOP_K','MIN_P','PRESENCE_PENALTY','REPETITION_PENALTY','MAX_NEW_TOKENS'})]:
        source=(root/'protocol'/filename).read_text();nodes=[];found=set()
        for node in ast.parse(source).body:
            assigned={t.id for t in node.targets if isinstance(t,ast.Name)} if isinstance(node,ast.Assign) else set()
            selected=({getattr(node,'name',None)}|assigned)&names
            if selected:nodes.append(node);found|=selected
        assert found==names
        exec(compile(ast.Module(body=nodes,type_ignores=[]),filename,'exec'),ns)
    ns.update(TEMPERATURE=.6,MAX_NEW_TOKENS=65536)
    assert ns['TOP_P']==.95 and ns['TOP_K']==20 and ns['MIN_P']==0 and ns['PRESENCE_PENALTY']==1.5 and ns['REPETITION_PENALTY']==1
    return SimpleNamespace(**{k:v for k,v in ns.items() if k!='__builtins__'})
