import importlib.util
from pathlib import Path
import sys
p=Path(__file__).parents[1]/"python_origin_audit.py"
s=importlib.util.spec_from_file_location("audit",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);ch=m.install()
if sys.argv[1]=="ok":
 import collections
 T=collections.namedtuple("Synthetic","x");assert eval("lambda z:z+1")(T(1).x)==2
else:
 raw=open('/tmp/ling_semantic_fixture_wh75l8c5/compile_support.txt',"rb").read();compile(raw,"<string>","exec")
