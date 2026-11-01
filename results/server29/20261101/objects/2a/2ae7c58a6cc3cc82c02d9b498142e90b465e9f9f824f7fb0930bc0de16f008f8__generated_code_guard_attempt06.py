"""Append-only generated Python provenance: observe exact inputs, never replace code.

Loader checks and Python compile/exec audit events are pre-execution boundaries.
Capture mode is engineering-only; verify mode cannot enlarge its manifest.
Roots are descriptive, never a prefix whitelist. Static identity is per file.
"""
from pathlib import Path
import hashlib, json, os, sys, time, types

def digest(data):return hashlib.sha256(data).hexdigest()
def filehash(path):return digest(Path(path).read_bytes())
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')

class GeneratedGuard:
    def __init__(self,static,output,mode='CAPTURE',manifest=None,packages=None):
        self.static={str(Path(p).resolve()):h for p,h in static.items()}
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False)
        self.mode=mode;assert mode in ('CAPTURE','VERIFY')
        self.allowed=(manifest or {}).get('modules',[])
        self.packages=packages or {}
        self.events=[];self.writers={};self.sources={};self.pending=[];self.executed={}
        self.active=False;self.inside=False;self.closed=False
        self.implementation=str(Path(__file__).resolve())
        self.started=time.time()
    def static_identity(self,path):
        p=Path(path);real=str(p.resolve())
        if real not in self.static:return False
        if not p.is_file() or filehash(p)!=self.static[real]:
            raise RuntimeError('STATIC_FROZEN_SOURCE_MISMATCH: '+str(p))
        return True
    def stack(self):
        rows=[];frame=sys._getframe(1)
        while frame:
            name=frame.f_code.co_filename
            if name!=self.implementation:
                rows.append(dict(module=frame.f_globals.get('__name__'),function=getattr(frame.f_code,'co_qualname',frame.f_code.co_name),path=name,line=frame.f_lineno))
            frame=frame.f_back
        return rows
    def producer(self,stack):
        # import machinery is not the generator; it cannot establish attribution.
        for row in stack:
            path=row['path'];module=row['module'] or ''
            if module.startswith(('importlib','_frozen_importlib')):continue
            if path.startswith('<'):continue
            if not self.static_identity(path):
                raise RuntimeError('UNKNOWN_GENERATED_PRODUCER: '+path)
            root=module.split('.')[0]
            version=self.packages.get(root)
            if version is None and '/site-packages/' not in path.replace('\\','/'):
                version='Python '+sys.version.split()[0]+'; exact frozen source SHA256'
            if version is None:
                raise RuntimeError('UNKNOWN_FROZEN_PRODUCER_PACKAGE_VERSION: '+module)
            return dict(producer_python_module=module,producer_qualified_function=row['function'],producer_source_path=path,producer_source_sha256=filehash(path),producer_package=root,producer_package_version=version,producer_callsite_line=row['line'])
        raise RuntimeError('UNATTRIBUTABLE_GENERATED_PRODUCER')
    @staticmethod
    def identity(record):
        # Random temporary paths, timestamps and stack addresses are audit-only.
        producer=record['producer']
        return dict(logical_module_name=record['logical_module_name'],source_sha256=record['source_sha256'],source_size=record['source_size'],producer={k:producer[k] for k in ['producer_python_module','producer_qualified_function','producer_source_sha256','producer_package','producer_package_version','producer_callsite_line']})
    def capture(self,name,source,path,producer,generation_stack,import_stack,event):
        if not isinstance(source,(str,bytes,bytearray)):
            raise RuntimeError('UNCAPTURABLE_GENERATED_SOURCE: '+name)
        data=source.encode('utf-8') if isinstance(source,str) else bytes(source)
        record=dict(logical_module_name=name,observed_temporary_path=path,source_size=len(data),source_sha256=digest(data),first_seen_timestamp=time.time(),first_import_or_exec_event=event,producer=producer,generation_call_stack=generation_stack,import_call_stack=import_stack,identity_class='GENERATED_EXECUTABLE_FROM_FROZEN_PRODUCER',source_representation='utf8_exact_input_str' if isinstance(source,str) else 'exact_input_bytes')
        copy=self.output/f'{len(self.events):06d}__{record["source_sha256"]}.py'
        with copy.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        record['canonical_captured_copy_path']=str(copy)
        identity=self.identity(record)
        if self.mode=='VERIFY' and identity not in self.allowed:
            dump(self.output/f'{len(self.events):06d}__REJECTED.json',record)
            raise RuntimeError('UNKNOWN_OR_CHANGED_GENERATED_EXECUTABLE_BEFORE_EXECUTION: '+name)
        self.events.append(record)
        dump(self.output/f'{len(self.events)-1:06d}__CAPTURE.json',record)
        return record
    def module_source(self,path,name):
        if self.static_identity(path):return 'STATIC_FROZEN_SOURCE'
        real=str(Path(path).resolve())
        if real in self.sources:
            assert filehash(path)==self.sources[real]['source_sha256'],'GENERATED_SOURCE_CHANGED_BEFORE_EXECUTION'
            return 'GENERATED_EXECUTABLE_FROM_FROZEN_PRODUCER'
        writer=self.writers.get(real)
        if writer is None:
            raise RuntimeError('UNKNOWN_STATIC_EXECUTABLE_NOT_OBSERVED_GENERATED: '+str(path))
        record=self.capture(name,Path(path).read_bytes(),str(path),writer['producer'],writer['stack'],self.stack(),'SourceFileLoader.get_code BEFORE_EXECUTION')
        self.sources[real]=record
        return 'GENERATED_EXECUTABLE_FROM_FROZEN_PRODUCER'
    def audit(self,event,args):
        if not self.active or self.inside or self.closed:return
        if event not in ('open','compile','exec'):return
        self.inside=True
        try:
            if event=='open':
                raw,mode,flags=args
                if not isinstance(raw,(str,bytes,os.PathLike)):return
                path=os.fsdecode(raw)
                # Only potential executable Python writes; not OS file ingress.
                writing=bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
                if not writing or not path.endswith('.py'):return
                real=str(Path(path).resolve())
                if real in self.static:raise RuntimeError('FROZEN_SOURCE_WRITE_FORBIDDEN: '+path)
                # Audit capture's own copies are not runtime executable inputs.
                if Path(real).is_relative_to(self.output):return
                stack=self.stack();self.writers[real]=dict(producer=self.producer(stack),stack=stack)
            elif event=='compile':
                source,filename=args;filename=os.fsdecode(filename)
                real=str(Path(filename).resolve()) if not filename.startswith('<') else filename
                if real in self.sources:
                    if source is None:raise RuntimeError('UNCAPTURABLE_GENERATED_AST: '+filename)
                    data=source.encode('utf8') if isinstance(source,str) else bytes(source)
                    assert digest(data)==self.sources[real]['source_sha256'],'GENERATED_COMPILE_INPUT_MISMATCH'
                    self.pending.append((filename,self.sources[real]['source_sha256']));return
                if real in self.static:
                    self.static_identity(filename)
                    if source is None:
                        # Existing explicitly frozen AST extraction helper semantics.
                        return
                    data=source.encode('utf8') if isinstance(source,str) else bytes(source)
                    if digest(data)==self.static[real]:return
                if source is None:raise RuntimeError('UNCAPTURABLE_DYNAMIC_AST_BEFORE_EXECUTION: '+filename)
                stack=self.stack();producer=self.producer(stack)
                if not filename.startswith('<'):
                    if real not in self.writers:raise RuntimeError('UNKNOWN_STATIC_SOURCE_COMPILE: '+filename)
                    producer=self.writers[real]['producer'];generation=self.writers[real]['stack']
                else:generation=stack
                name='dynamic::'+producer['producer_python_module']+'::'+producer['producer_qualified_function']+'::'+str(producer['producer_callsite_line'])+'::'+filename
                record=self.capture(name,source,filename,producer,generation,stack,'Python audit compile BEFORE_EXECUTION')
                self.sources[real]=record;self.pending.append((filename,record['source_sha256']))
            else:
                code=args[0];filename=code.co_filename
                real=str(Path(filename).resolve()) if not filename.startswith('<') else filename
                if real in self.static:self.static_identity(filename);return
                if id(code) not in self.executed:
                    index=next((i for i in range(len(self.pending)-1,-1,-1) if self.pending[i][0]==filename),None)
                    if index is None or real not in self.sources:
                        raise RuntimeError('UNCAPTURED_CODE_OBJECT_BEFORE_EXECUTION: '+filename)
                    _,source_hash=self.pending.pop(index)
                    assert source_hash==self.sources[real]['source_sha256'],'GENERATED_SOURCE_EXECUTION_ASSOCIATION_MISMATCH'
                    self.executed[id(code)]=(code,source_hash)
                # source already captured/verified at compile; exec itself checked too.
        finally:self.inside=False
    def install(self):
        sys.addaudithook(self.audit);self.active=True
    def finish(self):
        self.closed=True
        identities=[self.identity(r) for r in self.events]
        receipt=dict(mode=self.mode,modules=identities,generation_count=len(identities),events=self.events,preexec_capture=True,formal_measurements=0)
        dump(self.output/'GENERATED_CODE_RECEIPT.json',receipt)
        sums=''.join(r['source_sha256']+'  '+Path(r['canonical_captured_copy_path']).name+'\n' for r in self.events)
        with (self.output/'SHA256SUMS').open('x') as f:f.write(sums)
        return receipt

def deterministic_receipts(receipts):
    assert len(receipts)>=3
    normalized=[r['modules'] for r in receipts]
    # Sequence/multiplicity must match, not just membership.
    logical=[[m['logical_module_name'] for m in rows] for rows in normalized]
    if any(rows!=logical[0] for rows in logical[1:]):raise RuntimeError('GENERATED_MODULE_SET_DETERMINISM_GATE_FAIL')
    if any(rows!=normalized[0] for rows in normalized[1:]):raise RuntimeError('GENERATED_MODULE_BYTE_OR_PRODUCER_DETERMINISM_GATE_FAIL')
    return dict(GENERATED_MODULE_SET_DETERMINISM_GATE='PASS',GENERATED_MODULE_BYTE_DETERMINISM_GATE='PASS',GENERATION_COUNT_DETERMINISM_GATE='PASS',modules=normalized[0],generation_count=len(normalized[0]))
