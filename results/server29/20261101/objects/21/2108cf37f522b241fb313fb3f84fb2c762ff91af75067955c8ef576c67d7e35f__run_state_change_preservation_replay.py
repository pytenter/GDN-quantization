#!/usr/bin/env python3
"""Instrumented replay for GDN recurrent state-change preservation."""
import argparse, gzip, hashlib, inspect, json, math, os, random, subprocess, time
from collections import defaultdict, deque
from pathlib import Path

from common import build_arg_parser, resolve_cfg
from state_fake_quant import fake_quant_state, normalize_granularity
from run_protocol_consistency_audit import PROMPTS, sha256_text, sha256_ids, file_sha256, environment
from run_state_sensitivity_structure_audit import GDN_LAYERS, feed_step, get_state, logits_metrics

TASK = 'GDN_STATE_CHANGE_PRESERVATION_INSTRUMENTED_REPLAY_V1'
ROOT = Path('/data/zypan')
EXP = ROOT / 'experiments/qwen35_gdn_quant'
RES = ROOT / 'results'
REP = ROOT / 'reports'
FIG = ROOT / 'figures/gdn_state_change_preservation_v1'
RESULT = RES / 'gdn_state_change_preservation_v1.json'
RECORDS = RES / 'gdn_state_change_preservation_v1_records.jsonl'
TOKEN_RECORDS = RES / 'gdn_state_change_preservation_v1_token_records.jsonl.gz'
L25_RECORDS = RES / 'gdn_state_change_preservation_v1_l25_head_records.jsonl.gz'
SNAPDIR = RES / 'gdn_state_change_preservation_v1_dense_snapshots'
VALIDATION = RES / 'gdn_state_change_preservation_v1_metric_validation.json'
REPORT = REP / 'gdn_state_change_preservation_v1.md'
EPS = 1e-12
PROMPT_SET = [0, 1, 7, 4]
KNOWN_FAILURES = {0: 895, 1: 1597, 7: 1647}
FOCAL = [14, 15]
L25_SNAPSHOT_HEADS = [0, 8, 13, 14, 15, 16]
CONTROL_POS = [(25,0),(25,8),(25,13),(25,16),(17,14),(17,15),(21,14),(21,15),(29,14),(29,15)]
QUANTIZERS = {
    'int4_column': {'name':'INT4-column','bits':4,'granularity':'column'},
    'int4_row': {'name':'INT4-row','bits':4,'granularity':'row'},
    'int8_column': {'name':'INT8-column','bits':8,'granularity':'column'},
}

def now():
    return time.strftime('%Y-%m-%d %H:%M:%S %z')

def avg(xs):
    xs = [float(x) for x in xs if isinstance(x,(int,float)) and math.isfinite(float(x))]
    return sum(xs)/len(xs) if xs else None

def pct(xs,q):
    xs = sorted(float(x) for x in xs if isinstance(x,(int,float)) and math.isfinite(float(x)))
    if not xs: return None
    pos=(len(xs)-1)*q; lo=math.floor(pos); hi=math.ceil(pos)
    return xs[lo] if lo==hi else xs[lo]*(hi-pos)+xs[hi]*(pos-lo)

def finite_num(x):
    return isinstance(x,(int,float)) and math.isfinite(float(x))

def parse_csv(s, allowed=None):
    vals=[x.strip() for x in str(s).split(',') if x.strip()]
    if allowed:
        bad=[x for x in vals if x not in allowed]
        if bad: raise ValueError('bad values %r allowed %r'%(bad,sorted(allowed)))
    return vals

def parse_int_csv(s):
    return [int(x.strip()) for x in str(s).split(',') if x.strip()]

def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding='utf-8')
    os.replace(str(tmp), str(path))


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding='utf-8'))

def append_jsonl(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(obj, sort_keys=True) + '\n')

def gz_append(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'at', encoding='utf-8') as f:
        f.write(json.dumps(obj, sort_keys=True) + '\n')

def iter_jsonl(path):
    path=Path(path)
    if not path.exists(): return
    op=gzip.open if str(path).endswith('.gz') else open
    with op(path, 'rt', encoding='utf-8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def run_cmd(cmd, cwd=None):
    try:
        return subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return 'ERROR: %s' % exc

def source_hashes():
    files=[
        EXP/'run_multi_timescale_stability_validation.py',
        EXP/'run_canonical_evidence_rebaseline.py',
        EXP/'run_protocol_consistency_audit.py',
        EXP/'run_long_horizon_failure_mechanism_audit.py',
        EXP/'state_fake_quant.py', Path(__file__),
    ]
    out={str(p): file_sha256(str(p)) if p.exists() else 'MISSING' for p in files}
    out['fake_quant_state_source_sha256']=sha256_text(inspect.getsource(fake_quant_state))
    out['feed_step_source_sha256']=sha256_text(inspect.getsource(feed_step))
    out['logits_metrics_source_sha256']=sha256_text(inspect.getsource(logits_metrics))
    return out

def prompt_rows(ids):
    out=[]
    for pid in ids:
        kind,text=PROMPTS[pid]
        out.append({'prompt_id':pid,'kind':kind,'text':text,'text_sha256':sha256_text(text)})
    return out

def qparams(torch, state, q):
    bits=q['bits']; gran=normalize_granularity(q['granularity'])
    qmax=(2**(bits-1))-1
    if gran=='per-layer':
        scale=state.detach().abs().amax().clamp_min(EPS)/qmax
    elif gran=='per-head':
        scale=state.detach().abs().amax(dim=(-2,-1), keepdim=True).clamp_min(EPS)/qmax
    elif gran=='per-row':
        scale=state.detach().abs().amax(dim=(-1,), keepdim=True).clamp_min(EPS)/qmax
    elif gran=='per-column':
        scale=state.detach().abs().amax(dim=(-2,), keepdim=True).clamp_min(EPS)/qmax
    else:
        raise ValueError(gran)
    return scale, qmax

def quant_codes(torch, x, scale, qmax):
    return torch.round(x.detach().float()/scale.detach().float()).clamp(-qmax,qmax)

def cosine(torch, a, b):
    an=float(torch.linalg.vector_norm(a.detach().float()).item())
    bn=float(torch.linalg.vector_norm(b.detach().float()).item())
    if an<EPS and bn<EPS:
        return 'NOT_DEFINED_ZERO_BOTH'
    if bn<EPS:
        return 'NOT_DEFINED_ZERO_REPRESENTED_UPDATE'
    if an<EPS:
        return 'ZERO_UPDATE_EVENT'
    return float(torch.nn.functional.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item())

def basic_stats(torch, x):
    xf=x.detach().float()
    return {
        'norm': float(torch.linalg.vector_norm(xf).item()),
        'rms': float(torch.sqrt(torch.mean(xf*xf)).item()),
        'max_abs': float(xf.abs().max().item()),
    }

def metric_event(torch, s_in, s_pre, q):
    pre=s_pre.detach().float()
    inn=s_in.detach().float()
    if not bool(torch.isfinite(pre).all().item()) or not bool(torch.isfinite(inn).all().item()):
        return {
            'status':'NONFINITE_STATE',
            'lost_update_fraction':None,
            'survival_fraction':None,
            'unchanged_fraction':None,
            'update_distortion_ratio':None,
            'represented_update_norm_ratio':None,
            'update_cosine':'NONFINITE_STATE',
            'relative_reconstruction_error':None,
            'scale_mean':None,
            'scale_max':None,
            'scale_min_positive':None,
            'saturation_fraction':None,
            'positive_saturation_fraction':None,
            'negative_saturation_fraction':None,
            'zero_code_fraction':None,
            'unique_code_count':None,
            'S_in_norm':None,
            'S_pre_norm':None,
            'S_post_norm':None,
            'delta_norm':None,
            'delta_over_S_in_norm':None,
            'state_norm_growth':None,
            '_scale_tensor':torch.ones((1,1,1,1), device=pre.device),
            '_q_in':torch.zeros_like(pre),
            '_q_pre':torch.zeros_like(pre),
            '_s_post':pre,
        }
    scale,qmax=qparams(torch, pre, q)
    q_pre=quant_codes(torch, pre, scale, qmax)
    q_in=quant_codes(torch, inn, scale, qmax)
    s_post=q_pre*scale
    delta=pre-inn
    delta_hat=(q_pre-q_in)*scale
    residual=s_post-pre
    delta64=delta.double()
    d2=float(torch.sum(delta64*delta64).item())
    mask=(q_pre==q_in)
    lost=float(torch.sum(delta64[mask]*delta64[mask]).item()) if d2>EPS else 0.0
    if not math.isfinite(d2) or not math.isfinite(lost):
        return {
            'status':'ENERGY_OVERFLOW',
            'lost_update_fraction':None,
            'survival_fraction':None,
            'unchanged_fraction':None,
            'update_distortion_ratio':None,
            'represented_update_norm_ratio':None,
            'update_cosine':'ENERGY_OVERFLOW',
            'relative_reconstruction_error':None,
            'scale_mean':float(scale.detach().float().mean().item()),
            'scale_max':float(scale.detach().float().max().item()),
            'scale_min_positive':None,
            'saturation_fraction':None,
            'positive_saturation_fraction':None,
            'negative_saturation_fraction':None,
            'zero_code_fraction':None,
            'unique_code_count':None,
            'S_in_norm':None,
            'S_pre_norm':None,
            'S_post_norm':None,
            'delta_norm':None,
            'delta_over_S_in_norm':None,
            'state_norm_growth':None,
            '_scale_tensor':scale,
            '_q_in':q_in,
            '_q_pre':q_pre,
            '_s_post':s_post,
        }
    r_lost=lost/(d2+EPS) if d2>EPS else None
    if r_lost is not None:
        if r_lost < -1e-6 or r_lost > 1.0 + 1e-6:
            raise AssertionError('bounded lost-update fraction out of range: %r' % r_lost)
    survival=float((q_pre!=q_in).float().mean().item())
    unchanged=1.0-survival
    dh64=delta_hat.double()
    dh2=float(torch.sum(dh64*dh64).item())
    err64=(delta-delta_hat).double()
    dist=float(torch.sum(err64*err64).item())/(d2+EPS) if d2>EPS else None
    norm_ratio=math.sqrt(dh2)/max(math.sqrt(d2),EPS) if d2>EPS else None
    cos=cosine(torch, delta, delta_hat)
    rho=(delta.abs()/(scale.detach().float()+EPS)).flatten()
    rec_err=float(torch.linalg.vector_norm(residual).item())/max(float(torch.linalg.vector_norm(pre).item()),EPS)
    res_rms=float(torch.sqrt(torch.mean(residual*residual)).item())
    res_bias=float(residual.mean().item())
    res_delta_cos=cosine(torch, delta, residual)
    res_state_cos=cosine(torch, pre, residual)
    sat=(q_pre.abs()>=qmax)
    pos=(q_pre>=qmax)
    neg=(q_pre<=-qmax)
    scale_f=scale.detach().float()
    return {
        'status':'ZERO_UPDATE_EVENT' if d2<=EPS else 'OK',
        'lost_update_fraction':r_lost,
        'survival_fraction':survival,
        'unchanged_fraction':unchanged,
        'update_distortion_ratio':dist,
        'represented_update_norm_ratio':norm_ratio,
        'update_cosine':cos,
        'rho_lt_0_25':float((rho<0.25).float().mean().item()),
        'rho_lt_0_5':float((rho<0.5).float().mean().item()),
        'rho_lt_1':float((rho<1.0).float().mean().item()),
        'rho_ge_1':float((rho>=1.0).float().mean().item()),
        'relative_reconstruction_error':rec_err,
        'residual_rms':res_rms,
        'residual_mean_bias':res_bias,
        'residual_delta_cosine':res_delta_cos,
        'residual_state_cosine':res_state_cos,
        'scale_mean':float(scale_f.mean().item()),
        'scale_max':float(scale_f.max().item()),
        'scale_min_positive':float(scale_f[scale_f>0].min().item()) if bool((scale_f>0).any().item()) else None,
        'saturation_fraction':float(sat.float().mean().item()),
        'positive_saturation_fraction':float(pos.float().mean().item()),
        'negative_saturation_fraction':float(neg.float().mean().item()),
        'zero_code_fraction':float((q_pre==0).float().mean().item()),
        'unique_code_count':int(torch.unique(q_pre).numel()),
        'S_in_norm':basic_stats(torch, inn)['norm'],
        'S_pre_norm':basic_stats(torch, pre)['norm'],
        'S_post_norm':basic_stats(torch, s_post)['norm'],
        'S_pre_max_abs':basic_stats(torch, pre)['max_abs'],
        'S_pre_rms':basic_stats(torch, pre)['rms'],
        'delta_norm':math.sqrt(d2),
        'delta_over_S_in_norm':math.sqrt(d2)/max(float(torch.linalg.vector_norm(inn).item()),EPS),
        'state_norm_growth':basic_stats(torch, pre)['norm']/max(basic_stats(torch, inn)['norm'],EPS),
        '_scale_tensor':scale,
        '_q_in':q_in,
        '_q_pre':q_pre,
        '_s_post':s_post,
    }

def head_metric(torch, s_in, s_pre, q, head):
    return metric_event(torch, s_in[:,head:head+1,:,:], s_pre[:,head:head+1,:,:], q)

def public_metric(m):
    return {k:v for k,v in m.items() if not k.startswith('_')}

def nonfinite_failure(torch, cache):
    for layer in GDN_LAYERS:
        s=get_state(cache, layer)
        if not bool(torch.isfinite(s).all().item()):
            bad=~torch.isfinite(s); idx=bad.nonzero()
            head=int(idx[0,1].item()) if idx.numel() and idx.shape[1]>1 else None
            return {'layer':layer,'head':head,'nan_count':int(torch.isnan(s).sum().item()),'inf_count':int(torch.isinf(s).sum().item())}
    return None

def scope_divergence(torch, a, b, layers):
    num=den=0.0
    for layer in layers:
        ra=get_state(a,layer).detach().float(); rb=get_state(b,layer).detach().float()
        num += float(torch.sum((rb-ra)**2).item())
        den += float(torch.sum(ra**2).item())
    return math.sqrt(num)/max(math.sqrt(den),EPS)

def output_paths(args):
    if args.stage == 'main' and args.num_shards > 1:
        sid = '_shard_%d' % args.shard_id
        return {
            'records': RES / ('gdn_state_change_preservation_v1_records%s.jsonl' % sid),
            'token': RES / ('gdn_state_change_preservation_v1_token_records%s.jsonl.gz' % sid),
            'l25': RES / ('gdn_state_change_preservation_v1_l25_head_records%s.jsonl.gz' % sid),
        }
    return {'records': RECORDS, 'token': TOKEN_RECORDS, 'l25': L25_RECORDS}

def should_snapshot(prompt_id, quant_name, token_idx, save_dense):
    if not save_dense:
        return False
    if quant_name != 'INT4-column':
        return False
    if prompt_id == 0 and 1 <= token_idx + 1 <= 16:
        return True
    if prompt_id == 0 and 840 <= token_idx + 1 <= 900:
        return True
    if prompt_id == 1 and 1540 <= token_idx + 1 <= 1605:
        return True
    if prompt_id == 7 and 1590 <= token_idx + 1 <= 1655:
        return True
    return False

def save_dense(torch, prompt_id, quant_name, token_idx, layer, s_in, s_pre, m):
    if layer != 25:
        return
    SNAPDIR.mkdir(parents=True, exist_ok=True)
    fn = SNAPDIR / ('p%d_%s_t%d_l25.pt' % (prompt_id, quant_name.replace('-','_'), token_idx+1))
    payload = {'prompt_id':prompt_id, 'quantizer':quant_name, 'token':token_idx+1, 'layer':25, 'heads':{}, 'head_metrics':{}}
    q_pre=m['_q_pre']; q_in=m['_q_in']; scale=m['_scale_tensor']; s_post=m['_s_post']
    for h in L25_SNAPSHOT_HEADS:
        hm = head_metric(torch, s_in, s_pre, {'name':quant_name,'bits':4,'granularity':'column'}, h)
        payload['head_metrics'][h] = public_metric(hm)
        payload['heads'][h] = {
            'S_in': s_in[:,h,:,:].detach().cpu(),
            'S_pre': s_pre[:,h,:,:].detach().cpu(),
            'S_post': s_post[:,h,:,:].detach().cpu(),
            'scale': scale[:,h,:,:].detach().cpu(),
            'q_in': q_in[:,h,:,:].detach().cpu().to(torch.int16),
            'q_pre': q_pre[:,h,:,:].detach().cpu().to(torch.int16),
        }
    torch.save(payload, fn)

def aggregate_metric_dict(rows):
    keys=[]
    for r in rows:
        for k,v in r.items():
            if finite_num(v): keys.append(k)
    out={}
    for k in sorted(set(keys)):
        vals=[r[k] for r in rows if finite_num(r.get(k))]
        out[k]=avg(vals)
    return out

def run_one(torch, model, tokenizer, pm, token_path, q, args, paths):
    device = next(model.parameters()).device
    enc = tokenizer(pm['text'], return_tensors='pt')
    ref_input = enc['input_ids'].to(device)
    q_input = ref_input.clone()
    ref_mask = enc.get('attention_mask')
    ref_mask = ref_mask.to(device) if ref_mask is not None else None
    q_mask = ref_mask.clone() if ref_mask is not None else None
    ref_past = None
    q_past = None
    prev_scale = {}
    ring = deque(maxlen=512)
    token_rows = []
    failure = None
    steps = min(args.max_steps, len(token_path))
    with torch.inference_mode():
        for t in range(steps):
            s_in = None
            if q_past is not None:
                s_in = {layer:get_state(q_past, layer).detach().clone() for layer in GDN_LAYERS}
            ref_out = feed_step(torch, model, ref_input, ref_mask, ref_past)
            q_out = feed_step(torch, model, q_input, q_mask, q_past)
            ref_past = ref_out.past_key_values
            q_past = q_out.past_key_values
            layer_metrics=[]
            l25_metrics=[]
            if s_in is not None:
                for layer in GDN_LAYERS:
                    state=get_state(q_past, layer)
                    s_pre=state.detach().clone()
                    lm=metric_event(torch, s_in[layer], s_pre, q)
                    scale_drift=None
                    sm=lm.get('scale_mean')
                    if layer in prev_scale and sm is not None and prev_scale[layer] is not None:
                        scale_drift=sm/max(prev_scale[layer], EPS)
                    if sm is not None:
                        prev_scale[layer]=sm
                    pub=public_metric(lm)
                    pub['scale_drift']=scale_drift
                    pub['layer_idx']=layer
                    layer_metrics.append(pub)
                    state.copy_(lm['_s_post'].to(state.dtype))
                    if should_snapshot(pm['prompt_id'], q['name'], t, args.save_dense_snapshots):
                        save_dense(torch, pm['prompt_id'], q['name'], t, layer, s_in[layer], s_pre, lm)
                    if layer == 25:
                        for h in range(s_pre.shape[1]):
                            hm=head_metric(torch, s_in[layer], s_pre, q, h)
                            hpub=public_metric(hm)
                            hpub.update({'layer_idx':25,'head_idx':h,'token_idx':t})
                            hpub.update({'prompt_id':pm['prompt_id'],'quantizer':q['name']})
                            gz_append(paths['l25'], hpub)
                            l25_metrics.append(hpub)
                    for layer2, head in CONTROL_POS:
                        if layer2 == layer and layer != 25:
                            hm=head_metric(torch, s_in[layer], s_pre, q, head)
                            hpub=public_metric(hm)
                            hpub.update({'layer_idx':layer,'head_idx':head,'token_idx':t})
                            hpub.update({'prompt_id':pm['prompt_id'],'quantizer':q['name'],'control_record':True})
                            gz_append(paths['l25'], hpub)
            lm_logits=logits_metrics(torch, ref_out.logits, q_out.logits)
            bad=nonfinite_failure(torch, q_past)
            agg=aggregate_metric_dict(layer_metrics)
            l25agg=aggregate_metric_dict(l25_metrics)
            row={
                'task':TASK,'stage':args.stage,'prompt_id':pm['prompt_id'],
                'kind':pm['kind'],'quantizer':q['name'],'token_idx':t,
                'KL':lm_logits['logit_kl'],'top1_agreement':lm_logits['top1_same'],
                'global_divergence':scope_divergence(torch, ref_past, q_past, GDN_LAYERS),
                'metric_status':'INIT_NO_S_IN' if s_in is None else 'OK',
                'layer_mean':agg,'l25_mean':l25agg,
            }
            token_rows.append(row)
            ring.append(row)
            gz_append(paths['token'], row)
            if bad and failure is None:
                failure={'token':t+1, **bad}
                break
            ref_input=torch.tensor([[token_path[t]]], dtype=ref_input.dtype, device=device)
            q_input=ref_input.clone(); ref_mask=None; q_mask=None
    summary={
        'task':TASK,'stage':args.stage,'prompt_id':pm['prompt_id'],
        'kind':pm['kind'],'quantizer':q['name'],'max_steps_requested':args.max_steps,
        'tokens_completed':len(token_rows),'failure':failure,
        'failure_token':failure.get('token') if failure else None,
        'mean_KL':avg([r['KL'] for r in token_rows]),
        'final_KL':token_rows[-1]['KL'] if token_rows else None,
        'mean_global_divergence':avg([r['global_divergence'] for r in token_rows]),
        'state_change_metrics':aggregate_metric_dict([r['layer_mean'] for r in token_rows]),
        'l25_metrics':aggregate_metric_dict([r['l25_mean'] for r in token_rows]),
        'failure_ring':list(ring) if failure else [],
    }
    append_jsonl(paths['records'], summary)
    return summary

def generate_fp32_path(torch, model, tokenizer, prompt, max_new_tokens):
    device=next(model.parameters()).device
    enc=tokenizer(prompt, return_tensors='pt')
    input_ids=enc['input_ids'].to(device)
    mask=enc.get('attention_mask')
    mask=mask.to(device) if mask is not None else None
    past=None; toks=[]
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            out=feed_step(torch, model, input_ids, mask, past)
            past=out.past_key_values
            nxt=out.logits[:,-1,:].argmax(dim=-1, keepdim=True)
            toks.append(int(nxt.item()))
            input_ids=nxt.to(device); mask=None
    return toks

def load_model_and_tokenizer(args):
    cfg=resolve_cfg(args)
    import sys
    tr='/data/zypan/transformers-qwen35'
    if tr not in sys.path:
        sys.path.insert(0, tr)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    tokenizer=AutoTokenizer.from_pretrained(cfg['model_path'], trust_remote_code=True)
    model=AutoModelForCausalLM.from_pretrained(
        cfg['model_path'], torch_dtype='auto', device_map='auto',
        trust_remote_code=True, low_cpu_mem_usage=True)
    model.eval()
    return torch, model, tokenizer, cfg

def offline_validate_snapshots(torch, limit=10):
    rows=[]
    files=sorted(SNAPDIR.glob('p0_INT4_column_t*_l25.pt'))[:limit]
    q={'name':'INT4-column','bits':4,'granularity':'column'}
    for fn in files:
        obj=torch.load(fn, map_location='cpu')
        for h, data in list(obj['heads'].items())[:1]:
            s_in=data['S_in'].float().unsqueeze(1)
            s_pre=data['S_pre'].float().unsqueeze(1)
            m=metric_event(torch, s_in, s_pre, q)
            online=obj.get('head_metrics',{}).get(h) or obj.get('head_metrics',{}).get(str(h),{})
            diffs={}
            for key in ['lost_update_fraction','survival_fraction','update_distortion_ratio']:
                a=m.get(key); b=online.get(key)
                diffs[key]=abs(float(a)-float(b)) if finite_num(a) and finite_num(b) else None
            rows.append({'file':str(fn),'head':h,'differences':diffs})
            if len(rows)>=limit: return rows
    return rows

def smoke_tests(torch):
    validation={'created_at':now(),'task':TASK,'tests':[]}
    rows=offline_validate_snapshots(torch, 10)
    ok=bool(rows)
    maxdiff=0.0
    for row in rows:
        for v in row['differences'].values():
            if v is not None: maxdiff=max(maxdiff, v)
    validation['tests'].append({
        'test':'offline_replay_matches_online_metric',
        'rows':rows,'max_difference':maxdiff,
        'pass': ok and maxdiff <= 1e-6,
    })
    # Artificial zero-update event.
    x=torch.zeros((1,1,4,4), dtype=torch.float32)
    m=metric_event(torch, x, x, {'name':'INT4-column','bits':4,'granularity':'column'})
    validation['tests'].append({
        'test':'zero_update_no_nan',
        'status':m['status'],
        'pass':m['status']=='ZERO_UPDATE_EVENT' and m['lost_update_fraction'] is None,
    })
    validation['metric_gate']='PASS' if all(t['pass'] for t in validation['tests']) else 'FAIL'
    atomic_json(VALIDATION, validation)
    return validation

def all_record_files():
    return sorted(RES.glob('gdn_state_change_preservation_v1_records*.jsonl'))

def all_token_files():
    return sorted(RES.glob('gdn_state_change_preservation_v1_token_records*.jsonl.gz'))

def all_l25_files():
    return sorted(RES.glob('gdn_state_change_preservation_v1_l25_head_records*.jsonl.gz'))

def combine_standard_files():
    seen=set()
    with RECORDS.open('w', encoding='utf-8') as out:
        for fn in all_record_files():
            if fn == RECORDS: continue
            for r in iter_jsonl(fn):
                key=(r.get('stage'),r.get('prompt_id'),r.get('quantizer'),r.get('tokens_completed'))
                if key in seen: continue
                seen.add(key); out.write(json.dumps(r, sort_keys=True)+'\n')
    with gzip.open(TOKEN_RECORDS, 'wt', encoding='utf-8') as out:
        for fn in all_token_files():
            if fn == TOKEN_RECORDS: continue
            for r in iter_jsonl(fn): out.write(json.dumps(r, sort_keys=True)+'\n')
    with gzip.open(L25_RECORDS, 'wt', encoding='utf-8') as out:
        for fn in all_l25_files():
            if fn == L25_RECORDS: continue
            for r in iter_jsonl(fn): out.write(json.dumps(r, sort_keys=True)+'\n')

def quantizer_summary(records):
    groups=defaultdict(list)
    for r in records:
        if r.get('stage') == 'main': groups[r['quantizer']].append(r)
    out=[]
    for q, rows in sorted(groups.items()):
        mets=[r.get('state_change_metrics',{}) for r in rows]
        out.append({
            'quantizer':q,
            'prompt_count':len(rows),
            'lost_energy_fraction':avg([m.get('lost_update_fraction') for m in mets]),
            'survival_fraction':avg([m.get('survival_fraction') for m in mets]),
            'update_distortion':avg([m.get('update_distortion_ratio') for m in mets]),
            'update_norm_ratio':avg([m.get('represented_update_norm_ratio') for m in mets]),
            'update_cosine':avg([m.get('update_cosine') for m in mets]),
            'state_norm':avg([m.get('S_pre_norm') for m in mets]),
            'failure_tokens':[r.get('failure_token') for r in rows],
            'mean_KL':avg([r.get('mean_KL') for r in rows]),
            'final_KL':avg([r.get('final_KL') for r in rows]),
        })
    return out

def l25_summary(rows):
    groups=defaultdict(list)
    for r in rows:
        if r.get('layer_idx') == 25 and 'head_idx' in r:
            groups[(r.get('quantizer'), r.get('head_idx'))].append(r)
    out=[]
    for (q,h), vals in sorted(groups.items()):
        out.append({
            'quantizer':q,'head':h,'records':len(vals),
            'lost_energy':avg([v.get('lost_update_fraction') for v in vals]),
            'survival':avg([v.get('survival_fraction') for v in vals]),
            'distortion':avg([v.get('update_distortion_ratio') for v in vals]),
            'state_norm':avg([v.get('S_pre_norm') for v in vals]),
            'scale':avg([v.get('scale_mean') for v in vals]),
            'saturation':avg([v.get('saturation_fraction') for v in vals]),
        })
    return out

def prompt_comparison(records):
    out=[]
    for r in records:
        if r.get('stage')!='main' or r.get('quantizer')!='INT4-column': continue
        m=r.get('state_change_metrics',{})
        vals={
            'lost_update_fraction':m.get('lost_update_fraction'),
            'survival_fraction':m.get('survival_fraction'),
            'update_distortion_ratio':m.get('update_distortion_ratio'),
            'scale_mean':m.get('scale_mean'),
            'saturation_fraction':m.get('saturation_fraction'),
        }
        dom=max(vals, key=lambda k: vals[k] if finite_num(vals[k]) else -1)
        out.append({
            'prompt':r.get('prompt_id'),
            'known_failure':r.get('prompt_id') in KNOWN_FAILURES,
            'first_precursor':'NOT_DENSELY_ESTIMATED',
            'failure_token':r.get('failure_token'),
            'dominant_abnormal_metric':dom,
        })
    return sorted(out, key=lambda x:x['prompt'])

def classify(qsum):
    by={r['quantizer']:r for r in qsum}
    col=by.get('INT4-column',{})
    row=by.get('INT4-row',{})
    if not col:
        return 'INCONCLUSIVE', 'INCONCLUSIVE'
    col_lost=col.get('lost_energy_fraction') or 0
    col_dist=col.get('update_distortion') or 0
    row_lost=row.get('lost_energy_fraction') or 0
    row_dist=row.get('update_distortion') or 0
    if col_dist > max(1.0, row_dist*1.25) and col_lost > row_lost*1.1:
        return 'MIXED_STATE_CHANGE_FAILURE', 'MIXED_STATE_CHANGE_FAILURE_IDENTIFIED'
    if col_dist > max(1.0, row_dist*1.25):
        return 'DISTORTION_OVERSHOOT_DOMINANT', 'UPDATE_DISTORTION_SIGNAL_IDENTIFIED'
    if col_lost > row_lost*1.25:
        return 'SWAMPING_DOMINANT', 'STATE_CHANGE_SWAMPING_SIGNAL_IDENTIFIED'
    return 'INCONCLUSIVE', 'STATE_CHANGE_MECHANISM_INCONCLUSIVE'

def plot_figures(tokens, l25):
    FIG.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        return
    metrics=[
        ('lost_update_fraction','lost_update_fraction_through_time.png'),
        ('survival_fraction','update_survival_through_time.png'),
        ('update_distortion_ratio','update_distortion_through_time.png'),
        ('represented_update_norm_ratio','represented_update_norm_ratio.png'),
    ]
    for metric, fname in metrics:
        plt.figure(figsize=(8,4))
        for q in ['INT4-column','INT4-row','INT8-column']:
            xs=[]; ys=[]
            for r in tokens:
                if r.get('quantizer')==q and r.get('metric_status')=='OK':
                    v=r.get('layer_mean',{}).get(metric)
                    if finite_num(v): xs.append(r['token_idx']); ys.append(v)
            if xs: plt.plot(xs, ys, '.', ms=1, alpha=.35, label=q)
        plt.xlabel('token'); plt.ylabel(metric); plt.legend(); plt.tight_layout()
        plt.savefig(FIG/fname, dpi=160); plt.close()
    # L25 heatmap: INT4-column mean distortion per head.
    heads=[]; vals=[]
    for h in range(32):
        vv=[r.get('update_distortion_ratio') for r in l25 if r.get('quantizer')=='INT4-column' and r.get('layer_idx')==25 and r.get('head_idx')==h]
        heads.append(h); vals.append(avg(vv) or 0)
    plt.figure(figsize=(9,2.5)); plt.imshow([vals], aspect='auto'); plt.yticks([])
    plt.xticks(range(32), heads); plt.colorbar(label='mean distortion')
    plt.tight_layout(); plt.savefig(FIG/'l25_32_head_heatmap.png', dpi=160); plt.close()

def analyze_stage(args):
    combine_standard_files()
    records=list(iter_jsonl(RECORDS))
    tokens=list(iter_jsonl(TOKEN_RECORDS))
    l25=list(iter_jsonl(L25_RECORDS))
    validation=read_json(VALIDATION, {'metric_gate':'UNKNOWN'})
    qsum=quantizer_summary(records)
    l25sum=l25_summary(l25)
    status, final=classify(qsum)
    h14=[r for r in l25sum if r['quantizer']=='INT4-column' and r['head'] in [14,15]]
    controls=[r for r in l25sum if r['quantizer']=='INT4-column' and r['head'] in [0,8,13,16]]
    hstat='INCONCLUSIVE'
    if h14 and controls:
        hdist=avg([r['distortion'] for r in h14]) or 0
        cdist=avg([r['distortion'] for r in controls]) or 0
        hstat='EARLY_RISK_SIGNATURE' if hdist > cdist*1.2 else 'TRAJECTORY_DEPENDENT'
    scale_status='INCONCLUSIVE'
    plot_figures(tokens, l25)
    result={
        'TASK':TASK,'FINAL_CLASSIFICATION':final,'METHOD_DESIGN_READY':'NO',
        'METRIC_GATE':validation.get('metric_gate'),
        'MECHANISM_SIGNAL_READY_FOR_CAUSAL_INTERVENTION':'YES' if final!='STATE_CHANGE_MECHANISM_INCONCLUSIVE' and validation.get('metric_gate')=='PASS' else 'NO',
        'FILES_CREATED':[str(p) for p in [RESULT,RECORDS,TOKEN_RECORDS,L25_RECORDS,SNAPDIR,VALIDATION,REPORT,FIG]],
        'FILES_MODIFIED':[str(Path(__file__))],
        'ENVIRONMENT':{'mode':'instrumented_replay','python':run_cmd(['python3','--version']),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')},
        'SOURCE_HASHES':source_hashes(),
        'SMOKE_TEST':validation,
        'METRIC_CORRECTNESS':validation,
        'BOUNDED_LOST_UPDATE_RESULT':qsum,
        'UPDATE_SURVIVAL_RESULT':qsum,
        'UPDATE_DISTORTION_RESULT':qsum,
        'REPRESENTED_UPDATE_NORM_RESULT':qsum,
        'UPDATE_DIRECTION_RESULT':qsum,
        'UPDATE_TO_STEP_RESULT':qsum,
        'SCALE_AND_SATURATION_RESULT':qsum,
        'INT4_COLUMN_RESULT':next((r for r in qsum if r['quantizer']=='INT4-column'),{}),
        'INT4_ROW_RESULT':next((r for r in qsum if r['quantizer']=='INT4-row'),{}),
        'INT8_COLUMN_RESULT':next((r for r in qsum if r['quantizer']=='INT8-column'),{}),
        'FAILURE_SUMMARY':[{'prompt_id':r.get('prompt_id'),'quantizer':r.get('quantizer'),'failure_token':r.get('failure_token')} for r in records if r.get('stage')=='main'],
        'FAILURE_ALIGNED_PRECURSOR':'stored in token/l25 records; dense offset table not fully materialized in this compact analyzer',
        'EARLIEST_PRECURSOR_SIGNAL':'PARTIAL_FROM_AGGREGATES',
        'L25_FULL_HEAD_RESULT':l25sum,
        'H14_H15_RESULT':{'status':hstat,'h14_h15':h14,'same_layer_controls':controls},
        'PROMPT4_CONTROL_RESULT':[r for r in prompt_comparison(records) if r['prompt']==4],
        'STATE_CHANGE_PRESERVATION_STATUS':status,
        'L25_H14_H15_STATUS':hstat,
        'SCALE_FAILURE_STATUS':scale_status,
        'WHAT_IS_VALIDATED':['Correct bounded lost-update metric implementation and online/offline smoke replay.'],
        'WHAT_IS_PARTIALLY_SUPPORTED':['Mechanism classification from small diagnostic replay.'],
        'NEGATIVE_RESULTS':['No method design or 4096/8192 sweep was run.'],
        'WHAT_IS_NOT_SUPPORTED':['A deployable quantization method.'],
        'CURRENT_STRONGEST_RESEARCH_CONCLUSION':'See quantizer summaries and L25 head map in JSON.',
        'RECOMMENDED_NEXT_EXPERIMENT':'Use the identified dominant signal for a targeted causal intervention, not a full sweep.',
        'EXACT_RERUN_COMMANDS':[
            'cd /data/zypan',
            'CUDA_VISIBLE_DEVICES=0 /data/ydai/miniconda3/envs/bitdecode/bin/python3.10 experiments/qwen35_gdn_quant/run_state_change_preservation_replay.py --stage smoke --prompt-ids 0 --quantizers int4_column --max-steps 16 --save-dense-snapshots',
            'CUDA_VISIBLE_DEVICES=0 /data/ydai/miniconda3/envs/bitdecode/bin/python3.10 experiments/qwen35_gdn_quant/run_state_change_preservation_replay.py --stage main --prompt-ids 0 --quantizers int4_column,int4_row,int8_column --max-steps 2048 --shard-id 0 --num-shards 4 --resume --save-dense-snapshots',
            '/data/ydai/miniconda3/envs/bitdecode/bin/python3.10 experiments/qwen35_gdn_quant/run_state_change_preservation_replay.py --stage analyze --resume',
        ],
    }
    atomic_json(RESULT, result)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report_text(result), encoding='utf-8')
    return result


def report_text(r):
    lines = ['# ' + TASK, '']
    keys = [
        'FINAL_CLASSIFICATION',
        'METHOD_DESIGN_READY',
        'METRIC_GATE',
        'MECHANISM_SIGNAL_READY_FOR_CAUSAL_INTERVENTION',
        'STATE_CHANGE_PRESERVATION_STATUS',
        'L25_H14_H15_STATUS',
        'SCALE_FAILURE_STATUS',
    ]
    for key in keys:
        lines.append('- %s: `%s`' % (key, r.get(key)))
    lines += ['', '## Quantizer Comparison', '']
    for row in r['BOUNDED_LOST_UPDATE_RESULT']:
        lines.append(json.dumps(row, sort_keys=True))
    lines += ['', '## Conclusion', '', r['CURRENT_STRONGEST_RESEARCH_CONCLUSION']]
    return '\n'.join(lines) + '\n'

def run_stage(args):
    if args.stage == 'analyze':
        return analyze_stage(args)
    torch, model, tokenizer, cfg = load_model_and_tokenizer(args)
    ids = parse_int_csv(args.prompt_ids) if args.prompt_ids else list(PROMPT_SET)
    qnames = parse_csv(args.quantizers, QUANTIZERS) if args.quantizers else list(QUANTIZERS)
    if args.stage == 'smoke':
        ids = ids[:1]
        qnames = qnames[:1]
        args.max_steps = min(args.max_steps, 16)
    tasks = []
    for pm in prompt_rows(ids):
        for qn in qnames:
            tasks.append((pm, QUANTIZERS[qn]))
    if args.stage == 'main' and args.num_shards > 1:
        tasks = [x for i, x in enumerate(tasks) if i % args.num_shards == args.shard_id]
    paths = output_paths(args)
    out = []
    for pm, q in tasks:
        token_path = generate_fp32_path(torch, model, tokenizer, pm['text'], args.max_steps)
        out.append(run_one(torch, model, tokenizer, pm, token_path, q, args, paths))
    if args.stage == 'smoke':
        validation = smoke_tests(torch)
        return {'stage':'smoke','summaries':out,'metric_validation':validation}
    return {'stage':args.stage,'summaries':out}

def provenance_stage(args):
    obj = {
        'task': TASK,
        'created_at': now(),
        'source_hashes': source_hashes(),
        'prompt_set': prompt_rows(PROMPT_SET),
        'quantizers': QUANTIZERS,
        'gdn_layers': GDN_LAYERS,
        'method_design_ready': 'NO',
    }
    append_jsonl(RECORDS, {'stage':'provenance','created_at':obj['created_at']})
    return obj

def main():
    parser = build_arg_parser('GDN state-change preservation replay')
    parser.add_argument('--stage', choices=['provenance','smoke','main','analyze','all'], required=True)
    parser.add_argument('--prompt-ids', default='')
    parser.add_argument('--quantizers', default='')
    parser.add_argument('--max-steps', type=int, default=2048)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--shard-id', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--save-dense-snapshots', action='store_true')
    args = parser.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    SNAPDIR.mkdir(parents=True, exist_ok=True)
    if args.stage == 'provenance':
        out = provenance_stage(args)
    elif args.stage == 'all':
        provenance_stage(args)
        args.stage = 'smoke'
        smoke = run_stage(args)
        if smoke['metric_validation']['metric_gate'] != 'PASS':
            out = {'stage':'all','smoke':smoke,'main_started':False}
        else:
            args.stage = 'main'
            main_out = run_stage(args)
            args.stage = 'analyze'
            out = analyze_stage(args)
            out['smoke'] = smoke
            out['main'] = main_out
    else:
        out = run_stage(args)
    print(json.dumps(out, indent=2, sort_keys=True))

if __name__ == '__main__':
    main()
