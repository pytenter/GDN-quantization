"""Read-only FP teacher-forced observation of actual canonical KDA inputs."""
from common import *
from replay_core import *
import ast,argparse,gc,types,traceback

def qualified_runtime():
    # Invoke the exact inherited function in an isolated namespace, avoiding
    # importing a second module named common into this experiment's namespace.
    path=V4/'scripts/v4_common.py'
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='apply_qualified_runtime')
    ns={'os':os,'ast':ast,'Path':Path,'D':V4,'load':load,'c':types.SimpleNamespace(sha=sha)}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
    return ns['apply_qualified_runtime']()

class Capture:
    def __init__(self):
        runtime();qualified_runtime()
        self.P2=imp(PARENTS[2]/'scripts/phase2_final_hidden_512_smoke.py','surrogate_fp_phase2')
        original_rot=self.P2.load_rotations;original_tokens=self.P2.load_tokens
        self.P2.load_rotations=lambda args:{}
        self.P2.load_tokens=lambda args,tok:([],{'scope':'NO_MEASUREMENT'})
        try:
            rt,self.model,tok,self.probe,self.cap,_,_,_=self.P2.load_model_and_tools(argparse.Namespace(model=str(MODEL),legacy_repo=str(V3/'inputs/runtime_repo'),max_memory_gib=20))
        finally:self.P2.load_rotations=original_rot;self.P2.load_tokens=original_tokens
        assert not self.model.training and not any(p.requires_grad for p in self.model.parameters())
        assert all(p.device.type=='cuda' for p in self.model.parameters())
        self.device=self.P2.model_device(self.model);self.cap.enabled=False
        self.current={};self.post={};self.kernel=self.probe.orig_fused
        # Observe inside the inherited wrapper, AFTER canonical gate processing,
        # and call the same actual kernel with unchanged arguments.
        self.probe.globals['fused_recurrent_kda']=self.probe._wrap(self.observed_kernel)
        self.probe.globals['chunk_kda']=self.forbid_chunk

    def forbid_chunk(self,**kwargs):raise RuntimeError('CHUNK_PATH_FORBIDDEN: exact single-token fused recurrence required')

    def observed_kernel(self,**kwargs):
        layer=int(self.probe.current_layer)
        assert layer in LAYERS and layer not in self.current
        assert kwargs['q'].shape==(1,1,16,128) and kwargs['g'].shape==(1,1,16,128)
        assert kwargs['beta'].shape==(1,1,16)
        assert kwargs.get('use_qk_l2norm_in_kernel',False)
        assert not kwargs.get('use_gate_in_kernel',False)
        assert not kwargs.get('use_beta_sigmoid_in_kernel',False)
        assert not kwargs.get('state_v_first',False)
        assert kwargs['g'].dtype==torch.float32
        rec={k:kwargs[v].detach().clone() for k,v in [('q','q'),('k','k'),('v','v'),('beta','beta'),('log_decay','g')]}
        out,state=self.kernel(**kwargs)
        assert state.shape==(1,16,128,128) and state.dtype==torch.float32
        self.current[layer]=rec;self.post[layer]=state.detach()
        return out,state

    def step(self,cache,token,t):
        self.current={};self.post={};self.probe.begin('FP','native',None)
        try:
            with torch.inference_mode():
                out=self.model(input_ids=torch.tensor([[token]],device=self.device),attention_mask=torch.ones((1,t+1),device=self.device,dtype=torch.long),past_key_values=cache,cache_position=torch.tensor([t],device=self.device),use_cache=True)
        finally:self.probe.end()
        assert sorted(self.current)==LAYERS
        cache=out.past_key_values;del out
        for l in LAYERS:assert self.P2.get_state(cache,l).data_ptr()==self.post[l].data_ptr()
        return cache,self.current

    def document(self,ids,root,block=512,keep_step_hashes=False):
        assert not root.exists();root.mkdir(parents=True)
        cache=None;chunks=[];buf={l:{k:[] for k in ['q','k','v','beta','log_decay']} for l in LAYERS};history=[]
        started=time.time()
        for t,token in enumerate(ids):
            cache,rec=self.step(cache,token,t)
            for l in LAYERS:
                for k,v in rec[l].items():buf[l][k].append(v)
            if keep_step_hashes:history.append({str(l):tensor_sha(self.post[l]) for l in LAYERS})
            if (t+1)%block==0 or t+1==len(ids):
                chunk={k:torch.cat([torch.cat(buf[l][k],dim=1) for l in LAYERS],dim=0).cpu().contiguous() for k in buf[LAYERS[0]]}
                chunk['fp_after']=torch.cat([self.post[l] for l in LAYERS],dim=0).cpu().contiguous()
                assert all(torch.isfinite(v).all() for v in chunk.values())
                f=root/f'chunk_{len(chunks):03d}.pt';torch.save(chunk,f);f.chmod(0o444)
                chunks.append(dict(start_zero_based=t+1-chunk['q'].shape[1],end_one_based=t+1,artifact=row(f)))
                for l in LAYERS:
                    for k in buf[l]:buf[l][k].clear()
                del chunk,rec
                print('FP_TRACE_CAPTURE',root.name,t+1,'seconds',round(time.time()-started,2),flush=True)
        result=dict(gate='PASS',tokens=len(ids),layer_ids=LAYERS,chunks=chunks,initial_state='ALL_ZERO document start',fp_cache_dtype='torch.float32',record_layout='[layer-as-batch,token,true-head,key/value]',token_sha256=hashlib.sha256(b''.join(int(x).to_bytes(4,'little') for x in ids)).hexdigest(),elapsed_seconds=time.time()-started,step_hashes=history)
        emit(root/'trace_freeze.json',result)
        del cache;self.current={};self.post={};gc.collect();torch.cuda.empty_cache()
        return result

    def close(self):
        self.cap.close();self.probe.close();del self.model
        self.current={};self.post={};gc.collect();torch.cuda.empty_cache()

def canary(gpu):
    assert not (D/'run/DISCOVERY_STARTED.json').exists()
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(gpu);hardware()
    source=V3/'inputs/traces/wikitext2raw-train-000.pt'
    x=torch.load(source,map_location='cpu',weights_only=False);ids=[int(v) for v in x['token_ids'][:128]]
    root=D/'runtime'/f'capture_canary_gpu{gpu}'
    c=Capture()
    try:trace=c.document(ids,root,block=32,keep_step_hashes=True)
    finally:c.close()
    expected=load(V4/'runtime'/f'evaluation_gpu{gpu}_A.json')['numeric']['FP']
    assert len(expected)==4
    for t in range(4):assert trace['step_hashes'][t]==expected[t]['state'],('V4_FP_RUNTIME_BYTE_IDENTITY',gpu,t)
    state=torch.zeros((18,16,128,128),device='cuda',dtype=torch.float32)
    with torch.inference_mode():
        for entry in trace['chunks']:
            assert sha(entry['artifact']['path'])==entry['artifact']['sha256']
            x=torch.load(entry['artifact']['path'],map_location='cuda',weights_only=False)
            for t in range(x['q'].shape[1]):
                rec={k:x[k][:,t:t+1].contiguous() for k in ['q','k','v','beta','log_decay']}
                state=update_pre_q(state,rec)
            assert torch.equal(state,x['fp_after']),('CAPTURE_REPLAY_BITWISE_PARITY',entry['end_one_based'])
    emit(f'runtime/capture_canary_gpu{gpu}.json',dict(gate='PASS',physical_gpu=gpu,gpu_uuid=UUIDS[gpu],development_source=row(source),capture_manifest=row(root/'trace_freeze.json'),V4_FP_RUNTIME_BYTE_IDENTITY='PASS',CAPTURE_REPLAY_BITWISE_PARITY='PASS',no_fresh_documents=True))
    print('FP_CAPTURE_CANARY_PASS',gpu,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);a=p.parse_args()
    try:canary(a.gpu)
    except BaseException as e:
        emit(f'run/CAPTURE_CANARY_STOP_GPU{a.gpu}.json',dict(gate='FAIL',error=repr(e),traceback=traceback.format_exc(),no_auto_retry=True,scientific_measurements=0));raise
