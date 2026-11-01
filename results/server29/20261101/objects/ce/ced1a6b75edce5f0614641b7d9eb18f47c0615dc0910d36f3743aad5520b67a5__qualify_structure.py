"""CPU-only identity and fixed-shape structure audit; no model or GPU forward."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import argparse, ast, hashlib, json, math, pathlib, sys, time, traceback

EXPECTED_SEALS = {
    'inherited_cpu_v1': ('d659b34a8d8f7dc6196192bcad9d2f95a0aad3d9b1237f58f2e501777c58b015', 70),
    '.': ('ad024f3b4406680f1d4f7a6de6294f6f889c7667139b86f378bcec1b7ce9add2', 140),
    'execution_01': ('0afd31ce3992fc302556a88a3e6cb2f5e539850303efe824999c57ce7366ff54', 24),
    'execution_02': ('efece7e96e301f1a7bc40c66e1b89788a0d1387c17dd566083d7bf980b7a5af6', 43),
}
FORMAL = '/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/qwen_gdn/QWEN_C5_C6_AIME26_CANONICAL60_V1/scripts/run_qwen_c5_c6_formal.py'
CLUE = '/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/qwen_gdn/aime26/81920/run_qwen_aime26_formal.py'
ARCH = '/data/zypan/transformers-qwen35/src/transformers/models/qwen3_5/modeling_qwen3_5.py'
V3 = '/data/zypan/worktrees/qwen-gdn-deterministic-scale-discovery-v3/experiments/QWEN_GDN_DETERMINISTIC_RECOVERY_TO_SCALE_DISCOVERY_V3/scripts/numerical_recovery.py'

def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def segment(path, name):
    s = pathlib.Path(path).read_text()
    n = next(x for x in ast.parse(s).body if getattr(x, 'name', '') == name)
    return ast.get_source_segment(s, n)

def extract(path, names, namespace):
    tree = ast.parse(pathlib.Path(path).read_text())
    nodes = []
    for n in tree.body:
        if getattr(n, 'name', '') in names:
            # The original CPU helper body is retained verbatim. The Hub-dispatch
            # decorator is removed only to avoid loading any live runtime or GPU.
            if isinstance(n, ast.FunctionDef): n.decorator_list = []
            nodes.append(n)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace

def main(args):
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    p = pathlib.Path(args.parent)
    r = p / 'runtime_sources'
    result = dict(status='FAIL', scope='CPU_STRUCTURAL_AND_IDENTITY_ONLY', GPU_HOURS_USED=0,
                  model_loaded=False, model_forward=False, cuda_initialized=False,
                  not_a_full_model_native_H_equivalence_gate=True,
                  old_execution_02_status='FAIL_AND_STOP_PRESERVED', checks=[])
    try:
        seals = {}
        for label, (expected, count) in EXPECTED_SEALS.items():
            root = p / label
            manifest = root / 'hashes/artifact_sha256.txt'
            assert sha(manifest) == expected, ('seal manifest', label)
            lines = manifest.read_text().splitlines()
            assert len(lines) == count, ('seal count', label, len(lines))
            for line in lines:
                h, name = line.split('  ', 1)
                assert sha(root / name) == h, ('sealed bytes', label, name)
            seals[label] = dict(status='PASS_UNCHANGED', manifest_sha256=expected,
                                manifest_bytes=manifest.stat().st_size, verified_files=count,
                                receipt_sha256=sha(root/'hashes/DELIVERY_VERIFICATION.json'))
        result['seals'] = seals
        assert json.loads((p/'execution_02/FINAL_STATUS.json').read_text())['FINAL_CLASSIFICATION'] == 'FAIL_AND_STOP'
        sources = []
        for item in json.loads((p/'source_manifest.json').read_text())['files']:
            local = p / item['local_copy']
            assert sha(local) == item['sha256'], ('source copy', item['name'])
            row = dict(path=item['path'], local_copy=item['local_copy'], sha256=item['sha256'], copy_status='PASS')
            if args.live:
                assert sha(item['path']) == item['sha256'], ('live source', item['path'])
                row['live_status'] = 'PASS'
            sources.append(row)
        result['inherited_runtime_identity'] = sources
        assert sha(r/'formal_c128_runner.py') == '77b1e1f6dad8a0dd53a27cf5a47c02be86c26d2cf30732604ba6d3747e178f69'
        assert sha(r/'current_modeling_qwen3_5.py') == '90d929129ffc835d2652c604925c4f3842bc6e401e174ec6f0db2285dfb8f85a'
        assert sha(r/'numerical_recovery.py') == 'b13152f81ecd87428936dc5eced5a2a9f1e06ff992e8ba5bbb81703206dec26c'
        clue = pathlib.Path(CLUE) if args.live else pathlib.Path(__file__).parent/'source_clue_run_qwen_aime26_formal.py'
        same = []
        for n in ['hadamard','normalize_qk','QwenKeyHadamardPatch','get_state','quantize_c128']:
            a, b = segment(r/'formal_c128_runner.py',n), segment(clue,n)
            assert a == b, ('formal source clue function', n)
            same.append(dict(name=n, segment_sha256=hashlib.sha256(a.encode()).hexdigest(), byte_equal=True))
        result['formal_source_clue'] = dict(path=CLUE, sha256=sha(clue), segments=same,
                                           semantics_source=FORMAL)
        recurrent = segment(r/'current_modeling_qwen3_5.py','torch_recurrent_gated_delta_rule')
        block = segment(r/'current_modeling_qwen3_5.py','Qwen3_5GatedDeltaNet')
        formal = segment(r/'formal_c128_runner.py','QwenKeyHadamardPatch')
        qdq = segment(r/'formal_c128_runner.py','quantize_c128')
        old_science=(r/'scale_science.py').read_text()
        rules = [
            ('optional_qk_norm_before_FP32_Key_H128', 'value = normalize_qk(value) if use_norm else value\n        return value.float() @ hadamard(128, dtype=torch.float32, device=value.device)' in formal),
            ('Q_and_K_once_Value_unchanged', formal.count('query = self._transform(query, use_norm)') == 1 and formal.count('key = self._transform(key, use_norm)') == 1 and 'core, state = function(query, key, value,' in formal),
            ('no_second_kernel_normalization', 'use_qk_l2norm_in_kernel=False' in formal),
            ('real_recurrent_FP32_operands', 'contiguous().to(torch.float32)' in recurrent),
            ('unchanged_decay_delta_update_then_current_readout', recurrent.index('last_recurrent_state = last_recurrent_state * g_t') < recurrent.index('kv_mem =') < recurrent.index('last_recurrent_state = last_recurrent_state +') < recurrent.index('core_attn_out[:, :, i] =')),
            ('real_cache_stores_updated_state', 'cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)' in block),
            ('RMSNorm_gate_out_proj_original_body', 'self.norm(core_attn_out, z)' in block and 'self.out_proj(core_attn_out)' in block),
            ('canonical_C128_Key_axis_round_EPS_clamp_writeback', 'amax(dim=-2, keepdim=True).clamp_min(EPS) / 127.0' in qdq and 'torch.round(source / scale).clamp(-127, 127)' in qdq and 'state.copy_(dequant.to(state.dtype))' in qdq),
            ('qualified_fullmodel_forward_then_QDQ', old_science.index('out=self.model(input_ids=self.t.tensor([[token]]') < old_science.index('self.F.quantize_c128(cache,a)')),
            ('qualified_next_token_reads_QDQ', 'self.t.equal(initial,self.previous[c][l])' in old_science),
            ('V3_only_original_single_scan_substitution', "original.count('g = g.cumsum(dim=-1)') == 1" in segment(r/'numerical_recovery.py','chunk_binding') and "original.replace('g = g.cumsum(dim=-1)', 'g = v3_scan_backend(g, dim=-1)')" in segment(r/'numerical_recovery.py','chunk_binding')),
        ]
        result['static_semantics']=[dict(name=n,status='PASS' if ok else 'FAIL') for n,ok in rules]
        assert all(ok for _,ok in rules), rules
        result['compensated_scan'] = dict(source_path=V3, source_sha256=sha(r/'numerical_recovery.py'),
            function_sha256=hashlib.sha256(segment(r/'numerical_recovery.py','compensated_scan').encode()).hexdigest(),
            chunk_binding_sha256=hashlib.sha256(segment(r/'numerical_recovery.py','chunk_binding').encode()).hexdigest(),
            selected_route=json.loads((r/'NEW_NUMERICAL_CONTRACT.json').read_text())['selected_route'],
            unchanged=True)
        ns = extract(r/'formal_c128_runner.py', {'hadamard','normalize_qk','QwenKeyHadamardPatch'}, dict(torch=torch, math=math))
        arch = extract(r/'current_modeling_qwen3_5.py', {'l2norm','torch_recurrent_gated_delta_rule'}, dict(torch=torch,F=F))
        H = ns['hadamard'](128, dtype=torch.float32, device='cpu')
        ht = hashlib.sha256(H.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        assert ht == '6b407544270e0bc00aad4d734f52142c151b2e371447cfe93febe798cca55a68'
        I=torch.eye(128,dtype=torch.float32)
        orth=float((H.T@H-I).abs().max()); assert orth < 2e-6
        g=torch.Generator(device='cpu');g.manual_seed(20261015)
        state=torch.randn((1,32,128,128),generator=g,dtype=torch.float32)*.03
        rot=H.T@state; inv=H@rot
        inverse=float((inv-state).abs().max()); assert inverse < 2e-6
        q=torch.randn((1,4,32,128),generator=g,dtype=torch.float32).bfloat16()
        k=torch.randn((1,4,32,128),generator=g,dtype=torch.float32).bfloat16()
        v=torch.randn((1,4,32,128),generator=g,dtype=torch.float32)
        decay=-torch.rand((1,4,32),generator=g,dtype=torch.float32)*.15
        beta=torch.sigmoid(torch.randn((1,4,32),generator=g,dtype=torch.float32))
        patch=ns['QwenKeyHadamardPatch'](None)
        qn=ns['normalize_qk'](q);kn=ns['normalize_qk'](k)
        assert torch.equal(qn,arch['l2norm'](q,dim=-1,eps=1e-6))
        assert torch.equal(kn,arch['l2norm'](k,dim=-1,eps=1e-6))
        qh=patch._transform(q,True);kh=patch._transform(k,True)
        assert torch.equal(qh,qn.float()@H) and torch.equal(kh,kn.float()@H)
        assert not torch.equal(qh,qh@H)
        obs={}
        def mock(qq,kk,vv,*a,**kw):
            obs.update(query=torch.equal(qq,qh),key=torch.equal(kk,kh),value_same_object=vv is v,kernel_norm=kw['use_qk_l2norm_in_kernel'])
            return torch.zeros((1,4,32,128)),state
        patch.current_layer=0
        patch._wrap('recurrent',mock)(q,k,v,use_qk_l2norm_in_kernel=True)
        assert obs == dict(query=True,key=True,value_same_object=True,kernel_norm=False)
        # Use the actual unchanged source helper body, with original BF16 norm
        # already applied, to isolate the coordinate algebra from BF16 readout
        # truncation. Both paths use original FP32 recurrent arithmetic.
        core=arch['torch_recurrent_gated_delta_rule']
        out0,s0=core(qn.float(),kn.float(),v,decay,beta,state.clone(),True,False)
        outH,sH=core(qh,kh,v,decay,beta,rot.clone(),True,False)
        out_err=float((out0-outH).abs().max())
        st_err=float((s0-H@sH).abs().max())
        rel_out=float(torch.linalg.vector_norm((out0-outH).double())/torch.linalg.vector_norm(out0.double()))
        rel_state=float(torch.linalg.vector_norm((s0-H@sH).double())/torch.linalg.vector_norm(s0.double()))
        assert out_err < 4e-6 and st_err < 4e-6 and rel_out < 4e-6 and rel_state < 4e-6
        result['fixed_tensor_structure'] = dict(status='PASS',seed=20261015,state_shape=list(state.shape),
            query_key_shape=list(q.shape),original_input_dtype=str(q.dtype),recurrence_dtype='torch.float32',
            normalization_position='original BF16 before float32 H; native l2norm exact checked',
            matrix_sha256=ht,orthogonal_max_abs=orth,key_inverse_max_abs=inverse,
            readout_max_abs=out_err,state_inverse_max_abs=st_err,readout_relative_L2=rel_out,state_relative_L2=rel_state,
            fixed_tensor_acceptance='all max_abs and relative_L2 <4e-6; not a BF16 full-model equivalence threshold',
            wrapper_probe=obs,original_recurrent_body=True,compiled_decorators_removed_for_CPU_only=True,
            recurrence_source_sha256=hashlib.sha256(recurrent.encode()).hexdigest())
        prereg=json.loads((p/'preregistration.json').read_text())
        result['inherited_FP16_scale_safety'] = dict(source=str(p/'preregistration.json'),source_sha256=sha(p/'preregistration.json'),
            definition=prereg['scale_safety'], fixed_parameters=prereg['codec_parameters'] if 'codec_parameters' in prereg else None,
            floor=2**-24,ceiling=65504.,row_mean_floor=1e-20,LS_den_floor=1e-30,
            execution_union_rule='union of raw preclamp row and both initial/final column scale bound hits; mask intersects original S!=0',
            per_unit='doc/token/layer/head/codec',
            affected_energy_fraction='sum(S^2 on mask)/sum(S^2), FP64; 0 for total zero energy',
            affected_relative_MSE='sum((decode-S)^2 on mask)/sum(S^2 on mask), FP64; 0 for affected zero energy',
            severe_stop='affected energy fraction >=1e-4 AND affected relative MSE >1e-2',
            codec_source_sha256=sha(p/'inherited_cpu_v1/src/dual_axis_codec.py'))
        assert not torch.cuda.is_initialized()
        result['status']='PASS'
    except Exception as e:
        result['first_failure']=dict(error=repr(e),traceback=traceback.format_exc())
    result.update(cuda_initialized=torch.cuda.is_initialized(),torch=torch.__version__,python=sys.version,
                  script_sha256=sha(__file__),live_original_source_checks=bool(args.live),completed_unix=time.time())
    out=pathlib.Path(args.output)
    with out.open('x') as f: json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status=result['status'],cuda_initialized=result['cuda_initialized'],
                         seals={k:v['verified_files'] for k,v in result.get('seals',{}).items()},
                         first_failure=result.get('first_failure'),output=str(out))))
    if result['status']!='PASS':sys.exit(1)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--parent',required=True);a.add_argument('--output',required=True);a.add_argument('--live',action='store_true');main(a.parse_args())
