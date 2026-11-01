"""Synthetic-only architecture fixtures; isolated audit hooks per child."""
from pathlib import Path
import hashlib, importlib.util, json, subprocess, sys, tempfile
from generated_code_guard_attempt06 import GeneratedGuard, deterministic_receipts, filehash
def exercise(case):
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);out=root/'capture';p=root/'candidate.py';sentinel=root/'body_side_effect'
        source=b'VALUE = 7\n'
        static={str(Path(__file__).resolve()):filehash(__file__)}
        guard=GeneratedGuard(static,out)
        stack=guard.stack();producer=guard.producer(stack)
        def register(path):guard.writers[str(path.resolve())]=dict(producer=producer,stack=stack)
        def imported(path,name):
            guard.module_source(path,name)
            spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m);return m
        def denied(call):
            try:call()
            except (RuntimeError,AssertionError):return
            raise AssertionError('DID_NOT_FAIL_CLOSED')
        if case in ('T_GENERATED_01','T_GENERATED_02','T_GENERATED_08'):
            p.write_bytes(source);register(p);guard.install();m=imported(p,'fixture')
            assert m.VALUE==7 and guard.events and Path(guard.events[0]['canonical_captured_copy_path']).read_bytes()==source
            if case=='T_GENERATED_02':
                guard.active=False;q=root/'another_random_directory'/'candidate.py';q.parent.mkdir();q.write_bytes(source);register(q);guard.active=True
                n=imported(q,'fixture');assert n.VALUE==7
                assert guard.identity(guard.events[0])==guard.identity(guard.events[1])
            if case=='T_GENERATED_08':
                p.unlink();assert Path(guard.events[0]['canonical_captured_copy_path']).read_bytes()==source
        elif case in ('T_GENERATED_03','T_GENERATED_05','T_GENERATED_11'):
            p.write_bytes(source);register(p);guard.module_source(p,'fixture')
            identity=guard.identity(guard.events[0]);guard.mode='VERIFY';guard.allowed=[identity];guard.sources={}
            changed=b"from pathlib import Path\nPath("+repr(str(sentinel)).encode()+b").write_text('EXECUTED')\n"
            p.write_bytes(changed if case!='T_GENERATED_05' else source)
            guard.install();denied(lambda:imported(p,'unexpected' if case=='T_GENERATED_05' else 'fixture'))
            assert not sentinel.exists(),'BODY_RAN_BEFORE_REJECTION'
        elif case=='T_GENERATED_04':
            p.write_bytes(source);guard.install();denied(lambda:imported(p,'unknown'))
        elif case in ('T_GENERATED_06','T_GENERATED_07'):
            p.write_bytes(source);register(p);guard.module_source(p,'fixture');record=guard.identity(guard.events[0])
            rows=[json.loads(json.dumps(record)) for _ in range(3)]
            rows[2]['logical_module_name' if case=='T_GENERATED_06' else 'source_sha256']='changed'
            denied(lambda:deterministic_receipts([{'modules':[r]} for r in rows]))
        elif case=='T_GENERATED_09':
            p.write_bytes(source);guard.static[str(p.resolve())]=filehash(p);guard.install()
            assert imported(p,'ordinary').VALUE==7 and not guard.events
        elif case=='T_GENERATED_10':
            p=root/'locale_metadata';p.write_text('ambient');guard.install();assert p.read_text()=='ambient' and not guard.events
        elif case=='IMPORT_INTERPOSITION_SEMANTICS_GATE':
            # Identical globals/package/result and exception, with observer on/off.
            p.write_bytes(b'VALUE=sum([1,2,3]); PACKAGE=__package__; NAME=__name__\n');register(p)
            off=imported(p,'semantics');guard.install();on=imported(p,'semantics')
            assert (off.VALUE,off.PACKAGE,off.NAME)==(on.VALUE,on.PACKAGE,on.NAME)
            guard.active=False;p.write_bytes(b"raise ValueError('same exception')\n");guard.sources={};register(p)
            failures=[]
            for active in (False,True):
                guard.active=active
                try:imported(p,'exceptions')
                except ValueError as e:failures.append((type(e).__name__,str(e)))
            assert failures==[('ValueError','same exception')]*2
        elif case=='UNKNOWN_CODE_OBJECT_GATE':
            raw=compile('VALUE=9','<unobserved>','exec');guard.install();denied(lambda:exec(raw,{}))
        elif case=='FROZEN_PRODUCER_MUTATION_GATE':
            guard.static[str(Path(__file__).resolve())]='0'*64
            denied(lambda:guard.producer(stack))
        else:raise AssertionError(case)
        guard.finish()
    print(case,'PASS',flush=True)
def main():
    if len(sys.argv)>1:exercise(sys.argv[1]);return
    names=[f'T_GENERATED_{i:02d}' for i in range(1,12)]+['IMPORT_INTERPOSITION_SEMANTICS_GATE','UNKNOWN_CODE_OBJECT_GATE','FROZEN_PRODUCER_MUTATION_GATE']
    for case in names:subprocess.run([sys.executable,'-B',__file__,case],check=True)
    print('SYNTHETIC_TESTS_ALL_PASS',len(names),flush=True)
if __name__=='__main__':main()
