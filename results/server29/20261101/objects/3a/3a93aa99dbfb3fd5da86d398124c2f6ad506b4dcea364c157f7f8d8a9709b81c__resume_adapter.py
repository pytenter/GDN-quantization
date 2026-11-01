"""Independent continuation adapter; unchanged original token-loop body.

No model/Torch/CUDA import at module import or CPU AST audit. Runtime restoration
loads only the already SHA-bound Doc5 C6R_C128 checkpoint. Original source files,
failure receipts, completed trajectories and scientific seals are never edited.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path

PHASE_A='science_doc5_C6R_C128'
PHASE_D='science_doc5_C6R_DUAL_M'
CHECKPOINT_SHA='0d9c2e22a634b4327c03e34343087fd3d796276035f653b4b22ead3d8cf442ce'
PARENT_SOURCE_SHA='d76fcb4057eaaa781806ab6865e91998705ecb773449cf8851d2f0641969f9fc'
PARENT_PREREG_SHA='721038d6f9a48a7903eaa9339fe8823e95d5763e9b435b55199213625cbf5763'
DOC_ID='long_bench/repobench-p/test/b198a96effa37bfaebdf6b93b46a5d87a339bbda32162ab4'
INPUT_SHA='c53baae702b4c39e707f031102d63f9a50af2f81024deed8dbb0788f9450c3f3'
GPU_UUID='GPU-a78176ed-f97f-c54f-6201-5a01b570f422'


def need(ok,message):
    if not ok:raise RuntimeError(message)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def dump(node):return ast.dump(node,include_attributes=False)
def expr(source):return ast.parse(source,mode='eval').body


def build_run(qualified_run):
    """Change initialization, loop start and diagnostic counts; never its body."""
    original=copy.deepcopy(qualified_run);new=copy.deepcopy(original)
    withs=[x for x in new.body if isinstance(x,ast.With)]
    need(len(withs)==1 and len(withs[0].body)==1 and isinstance(withs[0].body[0],ast.For),'Original token loop layout changed')
    loop=withs[0].body[0];body_before=[dump(x) for x in loop.body]
    need(dump(loop.iter)==dump(expr('enumerate(ids)')),'Original enumerate contract changed')
    loop.iter=expr("enumerate(ids[continuation_meta['cursor']:], start=continuation_meta['cursor'])")
    changed={'cache_init':0,'score_init':0,'previous_seed':0,'call_count':0,'result_counts':0,'result_metadata':0}
    replacement=[]
    for node in new.body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            name=node.targets[0].id
            if name=='cache':
                need(isinstance(node.value,ast.Constant) and node.value.value is None,'Original cache initializer changed')
                replacement.extend(ast.parse('cache, prefix_scores, continuation_meta = _continuation_restore(f, condition, ids, doc_id, tag)').body);changed['cache_init']+=1;continue
            if name=='scores':
                need(isinstance(node.value,ast.Dict) and not node.value.keys,'Original scores initializer changed')
                node.value=expr('prefix_scores');changed['score_init']+=1
            if name=='result':
                need(isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='dict','Original result constructor changed')
                for keyword in node.value.keywords:
                    if keyword.arg=='writebacks':
                        keyword.value=expr("(len(ids)-continuation_meta['cursor'])*24 if condition!='FP_NATIVE' else 0");changed['result_counts']+=1
                    if keyword.arg=='next_token_consumption_checks':
                        keyword.value=expr("(len(ids)-continuation_meta['cursor']-(0 if continuation_meta['cursor'] else 1))*24 if condition!='FP_NATIVE' else 0");changed['result_counts']+=1
                replacement.append(node)
                replacement.extend(ast.parse("""result.update(resume_cursor=continuation_meta['cursor'],new_model_tokens=len(ids)-continuation_meta['cursor'],prefix_score_count=continuation_meta['prefix_score_count'],new_score_count=len(scores)-continuation_meta['prefix_score_count'],new_writebacks=result['writebacks'],new_next_token_consumption_checks=result['next_token_consumption_checks'],restore_audit_status=continuation_meta['restore_audit_status'],continuation=dict(continuation_meta,new_model_tokens=len(ids)-continuation_meta['cursor'],new_score_count=len(scores)-continuation_meta['prefix_score_count'],logical_consumed_tokens=len(ids),prefix_diagnostics='NOT_RETAINED' if continuation_meta['cursor'] else 'NOT_APPLICABLE',performance_scope='ONLY_NEW_SUFFIX_TOKENS' if continuation_meta['cursor'] else 'FULL_NEW_TRAJECTORY',mechanism_scope='ONLY_NEW_SUFFIX_SCORE_POSITIONS' if continuation_meta['cursor'] else 'FULL_NEW_TRAJECTORY',whole_trajectory_wall_seconds='NOT_RETAINED' if continuation_meta['cursor'] else result['wall_seconds'],allocator_peak_scope='CURRENT_PROCESS_ONLY'))""").body);changed['result_metadata']+=1;continue
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and dump(node.value.func)==dump(expr('K.check')):
            call=node.value
            if call.args and dump(call.args[0])==dump(expr('len(obs.calls)==len(ids)*24')):
                call.args[0]=expr("len(obs.calls)==(len(ids)-continuation_meta['cursor'])*24");changed['call_count']+=1
        replacement.append(node)
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='obs':
            replacement.extend(ast.parse("obs.previous={l:(f.get_state(cache,l).data_ptr(),K.tsha(f.get_state(cache,l))) for l in f.GDN_LAYERS} if continuation_meta['cursor'] else {}\n_continuation_record_restore(continuation_meta,obs.previous,tag)").body);changed['previous_seed']+=1
    new.body=replacement
    need(changed==dict(cache_init=1,score_init=1,previous_seed=1,call_count=1,result_counts=2,result_metadata=1),'Unexpected original run initialization/result layout: '+str(changed))
    need(body_before==[dump(x) for x in loop.body],'Scientific token-loop body changed')
    ast.fix_missing_locations(new)
    proof=dict(status='PASS_CPU_AST_ONLY',token_loop_body_AST_identical=True,original_loop_body_SHA256=hashlib.sha256('\n'.join(body_before).encode()).hexdigest(),original_run_AST_SHA256=hashlib.sha256(dump(original).encode()).hexdigest(),continuation_run_AST_SHA256=hashlib.sha256(dump(new).encode()).hexdigest(),allowed_changes=changed,no_GPU=True,does_not_establish_full32K_forward_error_bound=True)
    return new,proof


def build_install(qualified_install):
    """Keep the old context/codec/device/checkpoint wrappers; adapt count only."""
    node=copy.deepcopy(qualified_install);counts={'run_selector':0,'rotation_count':0}
    class Transform(ast.NodeTransformer):
        def visit_Call(self,n):
            if isinstance(n.func,ast.Name) and n.func.id=='adapted_run':
                n.func.id='_continuation_adapted_run';counts['run_selector']+=1
            return self.generic_visit(n)
        def visit_Assign(self,n):
            if len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='expected':
                need(dump(n.value)==dump(expr("result['consumed_tokens']*24")),'Original C6R rotation count expression changed')
                n.value=expr("result['new_model_tokens']*24");counts['rotation_count']+=1
            return self.generic_visit(n)
    node=Transform().visit(node)
    need(counts=={'run_selector':1,'rotation_count':1},'Original adapter install layout changed')
    ast.fix_missing_locations(node)
    return node,dict(status='PASS_CPU_AST_ONLY',changes=counts,original_install_AST_SHA256=hashlib.sha256(dump(qualified_install).encode()).hexdigest(),continuation_install_AST_SHA256=hashlib.sha256(dump(node).encode()).hexdigest(),C6R_class_codec_math_and_device_configuration_unchanged=True)


def bind_runtime(W,K,root,admission,phase):
    """Bind restore helpers after the exact original model loader has been wired."""
    root=Path(root)
    def restore(f,condition,ids,doc_id,tag):
        need(phase==tag and tag in (PHASE_A,PHASE_D),'Unregistered recovery trajectory')
        need(len(ids)==32768 and doc_id==DOC_ID,'Wrong recovery document/length')
        input_hash=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
        need(input_hash==INPUT_SHA,'Original Doc5 full input identity changed')
        common=dict(parent_source_manifest_sha256=PARENT_SOURCE_SHA,parent_preregistration_sha256=PARENT_PREREG_SHA,doc_id=doc_id,input_ids_sha256=input_hash,GPU_UUID=GPU_UUID)
        if phase==PHASE_D:
            need(condition=='C6R_DUAL_M','Fresh remaining arm condition changed')
            return None,{},dict(common,cursor=0,prefix_score_count=0,checkpoint_sha256=None,restore_audit_status='NOT_APPLICABLE_FRESH_FROM_TOKEN0')
        need(condition=='C6R_C128','Resumed condition changed')
        spec=admission['input_checkpoint'];p=root/spec['path']
        need(p.resolve().is_relative_to((root/'inputs').resolve()) and not p.is_symlink(),'Checkpoint escapes independent read-only input scope')
        need(spec['cursor']==16384 and spec['sha256']==CHECKPOINT_SHA and spec['bytes']==715984046,'Unexpected checkpoint admission')
        before=p.stat();need(before.st_size==spec['bytes'] and digest(p)==CHECKPOINT_SHA,'Restoration input SHA/bytes failed')
        payload=W.torch.load(p,map_location='cpu',weights_only=False)
        after=p.stat();need((before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns),'Checkpoint changed while loading')
        need(payload['cursor']==16384 and payload['doc_id']==doc_id and payload['condition']==payload['codec_identity']==condition,'Checkpoint cursor/doc/codec mismatch')
        need(payload['source_manifest_sha256']==PARENT_SOURCE_SHA and payload['prereg_sha256']==PARENT_PREREG_SHA,'Checkpoint frozen source/protocol mismatch')
        expected=[t for t in W.R.SCORE_POSITIONS if t<16384]
        scores=payload['completed_score_logits']
        need(payload['completed_score_positions']==list(scores)==expected and len(scores)==128,'Checkpoint prefix scores incomplete/reordered')
        need(payload['reducer_state']==dict(status='DEFERRED_CPU_FROM_SAVED_FULL_LOGITS',completed=expected),'Checkpoint deferred reducer state changed')
        for t,x in scores.items():
            need(x.device.type=='cpu' and x.dtype==W.torch.float32 and tuple(x.shape)==(1,1,248320) and bool(W.torch.isfinite(x).all()),'Checkpoint prefix full-vocab FP32 score mismatch')
        need(payload['cache'].get_seq_length()==16384,'Restored full-attention cache cursor mismatch')
        tensors=list(K.cache_tensors(payload['cache']))
        need(len(tensors)==64 and all(x.device.type=='cpu' and bool(W.torch.isfinite(x).all()) for _,x in tensors),'Checkpoint full cache tensor set invalid')
        cpu_hash=W.deep_hash(payload['cache'])
        cache=W.cpu_tree(payload['cache'],device='cuda:0')
        need(cache.get_seq_length()==16384,'GPU cache cursor changed')
        gpu_hash=W.deep_hash(cache)
        need(cpu_hash==gpu_hash,'CPU-to-GPU full cache bytes/metadata changed')
        for layer in f.GDN_LAYERS:
            x=f.get_state(cache,layer)
            need(x.dtype==W.torch.float32 and tuple(x.shape)==(1,32,128,128),'Restored recurrent QDQ state dtype/shape changed')
        W.torch.set_rng_state(payload['RNG_CPU']);W.torch.cuda.set_rng_state(payload['RNG_CUDA']);W.random.setstate(payload['RNG_python'])
        need(W.torch.equal(W.torch.get_rng_state(),payload['RNG_CPU']) and W.torch.equal(W.torch.cuda.get_rng_state(),payload['RNG_CUDA']) and W.random.getstate()==payload['RNG_python'],'Restored RNG mismatch')
        meta=dict(common,cursor=16384,prefix_score_count=128,checkpoint_sha256=CHECKPOINT_SHA,checkpoint_bytes=715984046,restore_audit_status='PASS',CPU_cache_deep_hash=cpu_hash,GPU_cache_deep_hash=gpu_hash,cache_tensor_count=64,RNG_restored_exact=True,first_new_token_index=16384,first_new_token_ID=int(ids[16384]),inherited_prefix_condition='C6R_C128',prefix_diagnostics='NOT_RETAINED')
        del payload,tensors
        return cache,scores,meta
    def record(meta,previous,tag):
        if not meta['cursor']:return
        need(len(previous)==24 and all(ptr>0 and isinstance(h,str) and len(h)==64 for ptr,h in previous.values()),'Restored next-token QDQ pointer/hash seed incomplete')
        K.save('analysis/RESTORE_'+tag+'.json',dict(meta,status='PASS',first_suffix_token_QDQ_previous_seeded=True,scope='Original CPU cache tree restored byte-for-byte to same physicalGPU; original next-kernel check remains active. No committed prefix forward repeated.'))
    W.__dict__.update(_continuation_restore=restore,_continuation_record_restore=record)
