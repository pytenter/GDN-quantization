"""One-time CPU source preparation; no engine imports or GPU operations."""
from pathlib import Path
p=Path(__file__).with_name('cublas_worker.py');s=p.read_text()
s=s.replace("TASK='QWEN_GDN_FAST_RUNTIME_QUALIFICATION_V1'", "TASK='QWEN_GDN_FAST_RUNTIME_QUALIFICATION_V1'\nRUN='QWEN_GDN_FAST_RUNTIME_QUALIFICATION_GPU_20261019_V1'\n_PROGRESS_PATH=None\n_PROGRESS_SEQ=0\n_SHARED_ROOT=None\ndef mark_progress(phase,token=None,bytes_completed=None):\n global _PROGRESS_SEQ\n if _SHARED_ROOT is not None:require(not(_SHARED_ROOT/'FIRST_FAILURE.json').exists(),'Shared sibling fail-closed STOP')\n if _PROGRESS_PATH is None:return\n _PROGRESS_PATH.parent.mkdir(parents=True,exist_ok=True);_PROGRESS_SEQ+=1\n kind='token_committed' if token is not None else 'bytes_hashed' if bytes_completed is not None else 'phase_transition'\n row=dict(sequence=_PROGRESS_SEQ,progress_kind=kind,phase=phase,token=token,bytes_completed=bytes_completed,monotonic_seconds=time.monotonic())\n with _PROGRESS_PATH.open('a') as f:f.write(json.dumps(row)+'\\n');f.flush()\n")
s=s.replace("for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)", "for i,b in enumerate(iter(lambda:f.read(4*1024**2),b'')):\n   h.update(b)\n   if i%16==15:mark_progress('hash:'+str(path),bytes_completed=(i+1)*4*1024**2)")
start=s.index('def transform_core(');end=s.index('\ndef profile_replay_ast',start)
s=s[:start]+'''def transform_core(source,workspace,physical_index=0):
 require(workspace in WORKSPACES.values() and type(physical_index)is int and physical_index in range(4),'Unregistered workspace/device')
 tree=ast.parse(source);env_nodes=[];visible_nodes=[]
 def visible_check(n):
  return isinstance(n,ast.Compare) and isinstance(n.left,ast.Call) and isinstance(n.left.func,ast.Attribute) and n.left.func.attr=='get' and ast.unparse(n.left.func.value)=='os.environ' and len(n.left.args)==1 and isinstance(n.left.args[0],ast.Constant) and n.left.args[0].value=='CUDA_VISIBLE_DEVICES' and len(n.comparators)==1
 for n in tree.body:
  if isinstance(n,ast.Assign)and len(n.targets)==1 and ast.unparse(n.targets[0])=="os.environ['CUBLAS_WORKSPACE_CONFIG']":
   require(isinstance(n.value,ast.Constant)and n.value.value==':4096:8','Original workspace literal changed');env_nodes.append(n);n.value=ast.copy_location(ast.Constant(workspace),n.value)
 for n in ast.walk(tree):
  if visible_check(n):
   require(isinstance(n.comparators[0],ast.Constant)and n.comparators[0].value=='0','Original physical visibility admission changed');visible_nodes.append(n);n.comparators[0]=ast.copy_location(ast.Constant(str(physical_index)),n.comparators[0])
 require(len(env_nodes)==len(visible_nodes)==1,'Exactly one workspace and physical-visibility control expected');ast.fix_missing_locations(tree)
 reverse=ast.parse(ast.unparse(tree))
 for n in reverse.body:
  if isinstance(n,ast.Assign)and len(n.targets)==1 and ast.unparse(n.targets[0])=="os.environ['CUBLAS_WORKSPACE_CONFIG']":n.value=ast.Constant(':4096:8')
 for n in ast.walk(reverse):
  if visible_check(n):n.comparators[0]=ast.Constant('0')
 require(ast.dump(reverse,include_attributes=False)==ast.dump(ast.parse(source),include_attributes=False),'Core AST changed outside two environment/device admission literals')
 return tree
''' +s[end:]
start=s.index('def controller_admission(');end=s.index('\ndef ambient_controls',start)
s=s[:start]+'''def controller_admission(root,authorization,auth,unit,plan,reader=None,parent_pid=None):
 require((root/'RUN_COPY_ORIGIN.json').is_file(),'GPU must use isolated authorized run copy')
 require(Path(plan['future_execution_root']).resolve()==root and plan['device']==auth['device'],'Unregistered card root/device')
 require(plan.get('no_GPU_time_limit')is True and auth.get('no_GPU_time_limit')is True and plan.get('progress_watchdog_seconds')==1800,'Human budget amendment changed')
 shared=Path(auth['shared_control_root']).resolve();require(not(shared/'FIRST_FAILURE.json').exists(),'Shared fail-closed STOP')
 meta=read(root/'execution/CONTROLLER_STARTED.json');dp=root/'execution'/unit['unit_id']/'DISPATCH.json';dispatch=read(dp)
 require(os.environ.get('QWEN_CONTROLLER_DISPATCH_SHA256')==sha(dp),'Missing immutable dispatch binding')
 pid=os.getppid()if parent_pid is None else parent_pid
 if reader is None:
  def reader(pid):
   f=Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split();a=Path('/proc/'+str(pid)+'/cmdline').read_bytes().rstrip(b'\\0').split(b'\\0');return int(f[19]),[x.decode()for x in a]
 tick,argv=reader(pid);script=str((root/'scripts/run_cublas_controller.py').resolve())
 require(pid==meta['controller_pid']==dispatch['controller_pid'] and tick==meta['controller_starttick']==dispatch['controller_starttick'],'Worker not direct child of admitted controller')
 require(argv==meta['controller_cmdline']and script in argv and '--root'in argv and Path(argv[argv.index('--root')+1]).resolve()==root,'Controller argv/root changed')
 require(meta['device']==dispatch['device']==auth['device'],'Controller/card device changed')
 require(meta['authorization_sha256']==dispatch['authorization_sha256']==sha(authorization)and meta['source_manifest_sha256']==dispatch['source_manifest_sha256']==auth['source_manifest_sha256'],'Controller source/authorization binding changed')
 require(dispatch['unit']==unit and dispatch.get('progress_policy')==dict(no_GPU_time_limit=True,watchdog_seconds=1800),'Finite unit/progress policy changed')
 return dispatch
''' +s[end:]
s=s.replace("require(a['device']['physical_index']==0 and a['device']['uuid']=='GPU-a78176ed-f97f-c54f-6201-5a01b570f422','Device authorization changed')", "require(a.get('execution_run_id')==RUN and type(a['device']['physical_index'])is int and a['device']['physical_index']in range(4)and a['device']['uuid'].startswith('GPU-'),'Device/run authorization changed')\n require(a.get('archive_config_sha256')==sha(root/'ARCHIVE_CONFIG.json')and a.get('human_authorization_sha256')==sha(root/'HUMAN_GPU_AUTHORIZATION.json')and a.get('amendment_sha256')==sha(root/'GPU_EXECUTION_AMENDMENT.json'),'Archive/human amendment identity changed')")
s=s.replace("controller_admission(root,args.authorization,a,unit,read(pp))", "controller_admission(root,args.authorization,a,unit,read(pp))\n global _PROGRESS_PATH,_SHARED_ROOT\n _PROGRESS_PATH=root/'execution'/unit['unit_id']/'PROGRESS.jsonl';_SHARED_ROOT=Path(a['shared_control_root']).resolve();mark_progress('source_admission')")
s=s.replace("fresh_environment(args.condition,{})", "fresh_environment(args.condition,{},device=a['device'])")
s=s.replace("def runtime(root,contract,condition):", "def runtime(root,contract,condition,device):")
s=s.replace("transform_core(src,WORKSPACES[condition])", "transform_core(src,WORKSPACES[condition],device['physical_index'])")
s=s.replace(" K.configure();torch=K.torch", " mark_progress('configure_runtime');K.configure();torch=K.torch")
s=s.replace("if x.split(', ')[0]=='0'", "if x.split(', ')[0]==str(device['physical_index'])")
s=s.replace("row[0][1]=='GPU-a78176ed-f97f-c54f-6201-5a01b570f422'", "row[0][1]==device['uuid']")
s=s.replace("own[0][1].strip()=='GPU-a78176ed-f97f-c54f-6201-5a01b570f422'", "own[0][1].strip()==device['uuid']")
s=s.replace(" model,qmod,env=K.load();", " mark_progress('model_load_started');model,qmod,env=K.load();mark_progress('model_load_complete');")
s=s.replace("env.update(CUBLAS_WORKSPACE_CONFIG=", "env.update(physical_device=device,logical_cuda_device=0,CUBLAS_WORKSPACE_CONFIG=")
s=s.replace("    row=store.record(t,rec.items,metadata)", "    row=store.record(t,rec.items,metadata);mark_progress('numerical_token_archived',token=t)")
s=s.replace("K,C,ns,base,model,qmod,env=runtime(root,contract,args.condition)", "K,C,ns,base,model,qmod,env=runtime(root,contract,args.condition,auth['device'])")
s=s.replace("  if args.mode=='qualify':", "  mark_progress('execution_'+args.mode)\n  if args.mode=='qualify':")
s=s.replace("only_core_AST_difference='Single module-level CUBLAS_WORKSPACE_CONFIG string literal; reverse AST exact'", "only_core_AST_difference='Workspace environment literal and physical CUDA visibility admission literal only; reverse AST exact; logicalcuda0 and math unchanged'")
s=s.replace("  libs=loaded_library_identity(contract);", "  mark_progress('execution_complete');libs=loaded_library_identity(contract);")
s=s.replace("  atomic_new(phase/'RESULT.json',result)", "  atomic_new(phase/'RESULT.json',result);mark_progress('result_committed')")
s=s.replace("   if not(root/'FIRST_FAILURE.json').exists():atomic_new(root/'FIRST_FAILURE.json',failure)", "   for fp in [root/'FIRST_FAILURE.json',Path(auth['shared_control_root'])/'FIRST_FAILURE.json']:\n    try:atomic_new(fp,failure)\n    except FileExistsError:pass")
p.write_text(s)
