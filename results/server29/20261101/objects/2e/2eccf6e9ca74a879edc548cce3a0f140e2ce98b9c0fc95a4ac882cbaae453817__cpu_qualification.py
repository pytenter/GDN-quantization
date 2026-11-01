"""One bounded CPU qualification. No CUDA, model loading, training experiment or retries."""
import os
if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise RuntimeError('CPU execution requires CUDA_VISIBLE_DEVICES empty')
from pathlib import Path
from types import SimpleNamespace
import ast, gc, hashlib, importlib, io, json, math, random, sys, time, traceback
import torch
import codec_training_adapter as A

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = Path('/data/zypan/analyses/qwen_gdn_c6r_dual_m_continuation_20261017/experiments/qwen_gdn/quantizer/QWEN_GDN_C6R_DUAL_M_INTERACTION_REUSE_5DOC_V1/checkpoints/CURRENT.pt')
CHECKPOINT_SHA = '99ecd34378ca21ccfdf4d4a352804ae1a45d43e63d4cce9d74084149cddd282c'
CHECKPOINT_BYTES = 1507209134
SEED = 20261017
CHECKS = 0
def need(ok, text):
    global CHECKS
    if not bool(ok):
        raise RuntimeError(text)
    CHECKS += 1
def bits(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.detach().contiguous().reshape(-1).view(torch.uint8), b.detach().contiguous().reshape(-1).view(torch.uint8))
def tree_hash(value):
    def describe(x):
        if isinstance(x, torch.Tensor):
            return ['tensor', str(x.dtype), list(x.shape), A.tensor_hash(x)]
        if isinstance(x, dict):
            return [[str(k),describe(v)] for k,v in sorted(x.items(), key=lambda p:str(p[0]))]
        if isinstance(x, (tuple,list)):
            return [describe(v) for v in x]
        return x
    return hashlib.sha256(json.dumps(describe(value),sort_keys=True,allow_nan=False).encode()).hexdigest()
def save(name, obj):
    with (ROOT/name).open('x') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')
def forbid_cuda(*args, **kwargs):
    raise RuntimeError('CPU qualification forbids any CUDA initialization')
def seed():
    torch.manual_seed(SEED)
    random.seed(SEED)
    if 'numpy' in sys.modules:
        sys.modules['numpy'].random.seed(SEED)

def fixtures(components):
    need(CHECKPOINT.stat().st_size == CHECKPOINT_BYTES and A.sha(CHECKPOINT) == CHECKPOINT_SHA, 'Actual-state checkpoint SHA/bytes')
    sys.path.insert(0, '/data/zypan/transformers-qwen35/src')
    payload = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    need(payload['cursor'] == 32768 and payload['condition'] == 'C6R_DUAL_M', 'Actual checkpoint identity')
    states = {i:components.base.get_state(payload['cache'],i).detach().clone() for i in components.core.GDN_LAYERS}
    for state in states.values():
        need(state.dtype == torch.float32 and tuple(state.shape) == (1,32,128,128) and bool(torch.isfinite(state).all()), 'Actual GDN shape/dtype')
    info = dict(source_path=str(CHECKPOINT),sha256=CHECKPOINT_SHA,bytes=CHECKPOINT_BYTES,cursor=payload['cursor'],condition=payload['condition'],doc_id=payload['doc_id'],
                state_sha256={str(i):A.tensor_hash(s) for i,s in states.items()},
                fixture_coordinates='Stored C6R+DualM coordinates; this is NOT a clean native FP teacher state',
                data_scope='HISTORICALLY_OBSERVED_DOC5; qualification fixture only; prohibited from new TRAIN/DEV/FUTURE test')
    del payload
    gc.collect()
    return states, info

def forward_and_gradient(components, actual):
    seed()
    cases = {
        'actual_32head': actual,
        'zero':torch.zeros_like(actual),
        'constant':torch.full_like(actual,.125),
        'tiny':torch.full_like(actual,1e-30),
        'seeded_random':torch.randn(actual.shape)*.1,
        'noncontiguous':actual.transpose(-1,-2),
    }
    sparse = torch.zeros_like(actual)
    sparse[0,0,0,0] = 1e10
    sparse[0,1,7,11] = -1e10
    cases['sparse_outlier'] = sparse
    records = []
    for name,x in cases.items():
        source = x.detach().clone().requires_grad_(True)
        reference = components.codec.encode(source,'DUAL_M_INT8')
        result = components.dual_m(source)
        for field in ('codes','rows','columns'):
            need(bits(getattr(reference,field),getattr(result,field)), 'D frozen '+name+' '+field)
        need(bits(reference.decode(),result.dequant), 'D hard reconstruction '+name)
        need(not result.codes.requires_grad and not result.rows.requires_grad and not result.columns.requires_grad, 'Detached metadata '+name)
        upstream = torch.randn_like(source)
        gradient = torch.autograd.grad(result.dequant,source,upstream)[0]
        need(bits(gradient,upstream), 'D identity STE '+name)
        legacy_input = x.detach().clone().requires_grad_(True)
        candidate_input = x.detach().clone().requires_grad_(True)
        legacy = components.legacy_c128(legacy_input)
        candidate = components.c128(candidate_input)
        need(all(bits(getattr(legacy,k),getattr(candidate,k)) for k in ('dequant','scale','codes')), 'A legacy bytes '+name)
        lg = torch.autograd.grad(legacy.dequant,legacy_input,upstream)[0]
        cg = torch.autograd.grad(candidate.dequant,candidate_input,upstream)[0]
        need(bits(lg,cg) and bits(cg,upstream), 'A legacy identity gradient '+name)
        hard_c128 = components.cayley.qwen_c128_qdq(x).dequant
        records.append(dict(name=name,shape=list(x.shape),D_code_SHA256=A.tensor_hash(result.codes),D_reconstruction_SHA256=A.tensor_hash(result.dequant),
                            A_legacy_vs_hard_different_FP32_values=int((candidate.dequant.detach()!=hard_c128).sum()),
                            A_legacy_vs_hard_byte_equal=bits(candidate.dequant,hard_c128)))
    # These call the original frozen helpers, rather than an approximate codec.
    rounding = torch.tensor([-127.5,-126.5,-1.5,-.5,0.,.5,1.5,126.5,127.5])
    expected = torch.tensor([-127.,-126.,-2.,-0.,0.,0.,2.,126.,127.])
    need(torch.equal(components.codec._original_levels(rounding),expected), 'Original ties-to-even/clamp')
    low, high = 2**-24, 65504.
    edges = torch.tensor([0.,low/2,low,float(torch.nextafter(torch.tensor(low),torch.tensor(float('inf')))),high,float(torch.nextafter(torch.tensor(high),torch.tensor(float('inf')))),high*2])
    scales = components.codec._original_safe(edges)
    need(scales.dtype == torch.float16 and bool(torch.isfinite(scales).all()) and float(scales.min())==low and float(scales.max())==high, 'Frozen FP16 scale bounds')
    midpoint = torch.tensor([1.+2**-11,1.+3*2**-11])
    need(torch.equal(components.codec._original_safe(midpoint),midpoint.half()), 'Frozen FP16 conversion ties')
    expected_rejections = []
    for value in (float('nan'),float('inf'),float('-inf')):
        bad = torch.zeros((1,1,128,128))
        bad[0,0,0,0] = value
        try:
            components.dual_m(bad)
        except (ValueError,RuntimeError,FloatingPointError) as exc:
            expected_rejections.append(type(exc).__name__)
        else:
            raise RuntimeError('Nonfinite input silently accepted')
    return dict(status='PASS',cases=records,fp16_scale_boundary='PASS',rounding_ties='PASS',expected_nonfinite_rejections=expected_rejections,
                policy='D true hard forward + whole-codec identity STE; scales/LS/round/clamp/FP16 detached; saturation is identity surrogate',
                A='Unchanged authoritative legacy qwen_c128_ste forward and gradient; any legacy hard-forward rounding mismatch disclosed')

def modules(components, states):
    norms, projections = {}, {}
    for i,state in states.items():
        norm = components.arch.Qwen3_5RMSNormGated(128).to(dtype=torch.bfloat16)
        projection = torch.nn.Linear(state.shape[1]*128,state.shape[1]*128,bias=False,dtype=torch.bfloat16)
        for parameter in list(norm.parameters())+list(projection.parameters()):
            parameter.requires_grad_(False)
        norms[i],projections[i] = norm,projection
    return norms,projections

def graph_and_updates(components, states):
    seed()
    # Full 24-layer parameter coverage, with a representative actual head per layer.
    selected = {i:s[:,:1].clone() for i,s in states.items()}
    smooth = components.cayley.CayleyDenseRotation()
    weight = torch.randn((128,128))*.1
    value = (smooth.matrix()*weight).sum()
    analytic = float(torch.autograd.grad(value,smooth.theta)[0][0])
    with torch.no_grad():
        smooth.theta[0]=.01
        plus=float((smooth.matrix()*weight).sum())
        smooth.theta[0]=-.01
        minus=float((smooth.matrix()*weight).sum())
        smooth.theta.zero_()
    finite_difference=(plus-minus)/.02
    need(abs(analytic-finite_difference)<=.002+.01*abs(analytic), 'Smooth Cayley directional finite difference')
    h=components.base.hadamard(128,dtype=torch.float32,device='cpu')
    fixture=selected[components.core.GDN_LAYERS[0]]
    delta=smooth.matrix()
    rotated=torch.einsum('ij,bhjv->bhiv',delta.T,torch.einsum('ij,bhjv->bhiv',h.T,fixture))
    restored=components.core.recover_state(rotated,delta,h)
    coordinate_relative_error=float(torch.linalg.vector_norm(restored-fixture)/torch.linalg.vector_norm(fixture).clamp_min(1e-12))
    need(coordinate_relative_error<=1e-4, 'Original key-coordinate recovery')
    norms, projections = modules(components, selected)
    coverage = []
    for arm in ('A','D'):
        seed()
        bank = components.cayley.PerLayerCayleyRotations(components.core.GDN_LAYERS)
        loss,info = A.local_diagnostic_loss(components,bank,selected,norms,projections,arm,tokens=3)
        need(bool(torch.isfinite(loss)), '24layer finite loss '+arm)
        loss.backward()
        layers=[]
        for i in components.core.GDN_LAYERS:
            grad=bank.layer(i).theta.grad
            need(grad is not None and bool(torch.isfinite(grad).all()) and bool((grad!=0).any()), '24layer gradient coverage '+arm)
            layers.append(dict(layer=i,gradient_norm=float(torch.linalg.vector_norm(grad))))
        need(info['QDQ_calls']==72 and info['next_QDQ_consumption_checks']==48, '24layer actual QDQ history')
        coverage.append(dict(arm=arm,loss=float(loss),state_loss=float(info['state_loss']),functional_loss=float(info['functional_loss']),layers=layers,
                             diagnostic_input_sha256=info['diagnostic_input_sha256'],teacher_target_sha256=info['teacher_target_sha256']))
    need(coverage[0]['diagnostic_input_sha256']==coverage[1]['diagnostic_input_sha256'] and coverage[0]['teacher_target_sha256']==coverage[1]['teacher_target_sha256'], 'A/D paired diagnostic inputs/teacher')
    # Legal H64 micro updates at optimizer boundaries, one representative layer.
    from training_state_io import save_legal_boundary_checkpoint, restore_legal_boundary_checkpoint
    micro=[]
    ids=(components.core.GDN_LAYERS[0],)
    local={i:selected[i] for i in ids}
    for arm in ('A','D'):
        seed()
        bank=components.cayley.PerLayerCayleyRotations(ids)
        optimizer=torch.optim.Adam(bank.parameters(),lr=.001,weight_decay=0)
        rows=[]
        context=dict(task='CPU_LOCAL_GRADIENT_QUALIFICATION_ONLY',arm=arm,codec={'A':'CANONICAL_C128','D':'DUAL_M_INT8'}[arm],
                     preregistration_sha256=A.sha(ROOT/'CPU_QUALIFICATION_PLAN.json'),
                     data_split_sha256=A.sha(ROOT/'qualification/ACTUAL_STATE_FIXTURE.json'),
                     teacher_manifest_sha256=A.sha(ROOT/'CPU_QUALIFICATION_PLAN.json'),
                     source_manifest_sha256=A.sha(ROOT/'CPU_SOURCE_MANIFEST.json'),
                     checkpoint_kind='MICRO_QUALIFICATION_ONLY',completed_updates=2,next_document_order_index=2,
                     completed_captures=2,captures_per_update=1,tokens_per_update=64,
                     boundary='AFTER_DOCUMENT_OPTIMIZER_STEP_CACHE_DISCARDED',
                     identity_scope='CPU_DIAGNOSTIC_UNIT_ONLY_NOT_FORMAL_PREREG_DATA_OR_TEACHER',
                     teacher_identity='SYNTHETIC_DIAGNOSTICS_ON_ACTUAL_CAPTURED_STATE',layer_roster=list(ids),horizon=64)
        checkpoint_path=ROOT/f'qualification/CPU_UNIT_{arm}_AFTER_UPDATE2.pt'
        def update(index):
            optimizer.zero_grad(set_to_none=True)
            python_rng_probe=random.random()
            started=time.monotonic()
            loss,info=A.local_diagnostic_loss(components,bank,local,norms,projections,arm,tokens=64)
            need(bool(torch.isfinite(loss)), 'Micro finite loss')
            loss.backward()
            parameters=list(bank.parameters())
            need(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters), 'Micro finite gradients')
            norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
            need(float(norm)>0 and math.isfinite(float(norm)), 'Micro nonzero gradient')
            optimizer.step()
            orth=bank.runtime_gate(1e-4)
            need(orth['status']=='PASS', 'Original orthogonality threshold')
            need(info['QDQ_calls']==64 and info['next_QDQ_consumption_checks']==63, 'H64 hard history consumption')
            return dict(update=index,loss=float(loss),state_loss=float(info['state_loss']),functional_loss=float(info['functional_loss']),
                        gradient_norm=float(norm),orthogonality_error=orth['max_abs_rt_r_minus_i'],
                        bank_sha256=tree_hash(bank.state_dict()),optimizer_sha256=tree_hash(optimizer.state_dict()),
                        diagnostic_input_sha256=info['diagnostic_input_sha256'],teacher_target_sha256=info['teacher_target_sha256'],
                        python_rng_probe=python_rng_probe,CPU_seconds=time.monotonic()-started,
                        tokens_per_optimizer_microstep=64,full_model=False)
        rows.append(update(1));rows.append(update(2))
        saved=save_legal_boundary_checkpoint(checkpoint_path,bank,optimizer,torch,context=context)
        rows.append(update(3))
        restored=restore_legal_boundary_checkpoint(checkpoint_path,bank,optimizer,torch,expected_context=context,expected_sha256=saved['sha256'])
        replay=update(3)
        for key in ('loss','state_loss','functional_loss','gradient_norm','orthogonality_error','bank_sha256','optimizer_sha256','diagnostic_input_sha256','teacher_target_sha256','python_rng_probe'):
            need(rows[2][key]==replay[key], 'Exact resumed third step '+key)
        micro.append(dict(arm=arm,updates=rows,resumed_update3=replay,checkpoint_saved=saved,checkpoint_restored=restored,
                          status='PASS_EXACT_LEGAL_OPTIMIZER_BOUNDARY_RESUME',scope='One representative layer/head with real captured state; original H64 and loss helper; not a full1024-token/8-capture update'))
    return dict(status='PASS_LOCAL_CPU_ONLY',coverage_24layers=coverage,three_H64_micro_updates=micro,
                smooth_Cayley_check=dict(analytic=analytic,finite_difference=finite_difference,tolerance='.002+.01*abs(analytic); FP32 original solve, no hard-rounding finite difference'),
                key_coordinate_relative_error=coordinate_relative_error,
                full_model_training='NOT_TESTED',diagnostic_norm_and_projection_weights='SEEDED_FROZEN_UNIT_WEIGHTS_NOT_MODEL_WEIGHTS',
                diagnostic_teacher_scope='The captured C6R-coordinate tensor is a numerical fixture; no claim it is the historical clean FP teacher state',
                functional_scope='Original RMSNorm-before-gate class + local projection + original out_proj hook/captured_losses AST; helper qualification, not real9B functional trajectory')

def main():
    if (ROOT/'FIRST_CPU_FAILURE.json').exists() or (ROOT/'CPU_QUALIFICATION_STARTED.json').exists():
        raise RuntimeError('Prior CPU attempt exists; no automatic retry')
    need(not torch.cuda.is_initialized(), 'CUDA already initialized')
    torch.cuda._lazy_init=forbid_cuda
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest=json.loads((ROOT/'CPU_SOURCE_MANIFEST.json').read_text())
    for rel,digest in manifest['execution_sources'].items():
        need(A.sha(ROOT/rel)==digest, 'Pre-execution source SHA mismatch '+rel)
    save('CPU_QUALIFICATION_STARTED.json',dict(status='CPU_ONLY_STARTED',pid=os.getpid(),source_manifest_SHA256=A.sha(ROOT/'CPU_SOURCE_MANIFEST.json'),CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES']))
    try:
        components=A.QualifiedComponents()
        states,actual=fixtures(components)
        (ROOT/'qualification').mkdir(exist_ok=True)
        save('qualification/ACTUAL_STATE_FIXTURE.json',actual)
        forward=forward_and_gradient(components,states[components.core.GDN_LAYERS[0]])
        save('DUAL_M_FORWARD_PARITY.json',dict(forward,actual_state_provenance=actual,no_GPU=True,source_hashes=components.source_hashes))
        graph=graph_and_updates(components,states)
        need(not torch.cuda.is_initialized(), 'CPU run initialized CUDA')
        save('GRADIENT_QUALIFICATION_CPU.json',dict(graph,checks=CHECKS,CUDA_initialized=torch.cuda.is_initialized(),AST_provenance=components.AST_provenance,
              original_C128_forward='UNCHANGED_LEGACY_STE',DualM_forward='FROZEN_CODEC_BIT_EXACT',backward='WHOLE_CODEC_IDENTITY_STE_SCALES_DETACHED',
              actual_state_provenance=actual,source_manifest_SHA256=A.sha(ROOT/'CPU_SOURCE_MANIFEST.json')))
        print(json.dumps(dict(status='PASS_CPU_LOCAL_QUALIFICATION',checks=CHECKS,CUDA_initialized=False,full_model_training='NOT_TESTED')))
    except BaseException as exc:
        save('FIRST_CPU_FAILURE.json',dict(status='FAIL_CLOSED_STOP',error=repr(exc),traceback=traceback.format_exc(),CUDA_initialized=torch.cuda.is_initialized()))
        raise

if __name__=='__main__':
    main()
