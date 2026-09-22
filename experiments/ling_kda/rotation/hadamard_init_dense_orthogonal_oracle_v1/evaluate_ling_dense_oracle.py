#!/usr/bin/env python3
"""Held-out local and persistent evaluation for the Ling dense oracle."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import statistics
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
    return module


R = import_file(HERE / "run_ling_dense_oracle.py", "ling_dense_oracle_runner_for_eval")
HORIZONS = (1, 4, 8, 16, 32, 64, 128)


def load_bank(path, layers, device):
    payload = torch.load(path, map_location="cpu")
    bank = R.C.PerLayerCayleyRotations(layers).to(device); bank.load_state_dict(payload["bank"]); bank.eval()
    return bank, payload


def matrices(method, banks, layers, device):
    identity = torch.eye(128, device=device, dtype=torch.float32)
    if method in ("Native_INT8", "Hadamard"): return {layer: identity for layer in layers}
    return {layer: banks[method].layer(layer).matrix().detach() for layer in layers}


def rotated(method): return method != "Native_INT8"


def local_method(model, probe, layers, traces, method, banks, h, device):
    mats = matrices(method, banks, layers, device)
    accum = {key: [] for key in ("state_error", "core_error", "post_norm_error", "post_gate_error", "out_proj_error", "saturation", "scale_mean", "scale_max")}
    with torch.no_grad():
        for sample in traces:
            per = {key: [] for key in accum}
            for layer in layers:
                rec = R.device_record(sample["layers"][layer], device); state = rec["state_input"]
                matrix = mats[layer]
                if rotated(method):
                    state_rot = R.transform_state(state, matrix, h).to(torch.bfloat16)
                    exact = R.C.ling_r128_qdq(state_rot.float())
                    qdq = exact.dequant.to(torch.bfloat16).float()
                    recovered = R.recover_state(qdq, matrix, h)
                    value = rec["v"].float().matmul(h).matmul(matrix).to(torch.bfloat16)
                    raw, _ = R.fused_replay(probe, rec, qdq, value, device)
                    core = raw.float().matmul(matrix.T).matmul(h).to(torch.bfloat16)
                else:
                    exact = R.C.ling_r128_qdq(state.float())
                    qdq = exact.dequant.to(state.dtype).float()
                    recovered = qdq
                    core, _ = R.fused_replay(probe, rec, qdq, rec["v"], device)
                norm, output = R.local_output(model, layer, core, rec["dynamic_gate"])
                per["state_error"].append(float(R.C.relative_mse(recovered, state.float()).cpu()))
                per["core_error"].append(float(R.C.relative_mse(core, rec["core_output"]).cpu()))
                per["post_norm_error"].append(float(R.C.relative_mse(norm, rec["post_norm_gate"]).cpu()))
                per["post_gate_error"].append(per["post_norm_error"][-1])
                per["out_proj_error"].append(float(R.C.relative_mse(output, rec["out_proj_output"]).cpu()))
                per["saturation"].append(float((exact.codes.abs() == 127).float().mean().cpu()))
                per["scale_mean"].append(float(exact.scale.mean().cpu())); per["scale_max"].append(float(exact.scale.max().cpu()))
            for key in accum: accum[key].append(sum(per[key]) / len(per[key]))
    return {key: sum(values) / len(values) for key, values in accum.items()} | {"samples": len(traces)}


class PerLayerHistoryProbe:
    """Factory wrapper because the canonical probe expects one global matrix."""
    @staticmethod
    def build(F, mats):
        parent = F.H.HistoryFormationProbe
        class Dynamic(parent):
            def __init__(self):
                self._fallback_rotation = torch.eye(128, dtype=torch.float32)
                self.layer_matrices = mats
                super().__init__(self._fallback_rotation)
            @property
            def rotation(self):
                layer = getattr(self, "current_layer", None)
                return self.layer_matrices.get(int(layer), self._fallback_rotation) if layer is not None else self._fallback_rotation
            @rotation.setter
            def rotation(self, value): self._fallback_rotation = value
        return Dynamic()


def quantize_cache(F, cache, layers):
    stack = F.BASE.cache_stack(cache, layers)
    qdq = {layer: R.C.ling_r128_qdq(stack[layer].float()).dequant.to(stack[layer].dtype) for layer in layers}
    F.H.replace_stack(cache, qdq)


def snapshot(F, cache, layers):
    stack = F.BASE.cache_stack(cache, layers)
    return {layer: stack[layer].detach().float().cpu().clone() for layer in layers}


def recovered_snapshot(F, cache, layers, method, mats, h):
    stack = F.BASE.cache_stack(cache, layers); output = {}
    for layer in layers:
        # cache_stack is an audit accessor and may return CPU snapshots even
        # while the live cache and per-layer correction are CUDA-resident.
        # Device transfer here is read-only metric plumbing; it does not write
        # back to the runtime cache or alter the deployed numerical path.
        value = stack[layer].detach().float().to(mats[layer].device)
        output[layer] = R.recover_state(value, mats[layer], h).cpu() if rotated(method) else value.cpu()
    return output


def state_rel(current, reference, layers):
    return sum(float(torch.linalg.vector_norm(current[layer] - reference[layer]) / torch.linalg.vector_norm(reference[layer]).clamp_min(R.EPS)) for layer in layers) / len(layers)


def logit_metrics(value, reference):
    x, y = value.float(), reference.float(); logp, logq = torch.log_softmax(y, -1), torch.log_softmax(x, -1)
    kl = float((logp.exp() * (logp - logq)).sum().cpu())
    rel = float((torch.linalg.vector_norm(x-y) / torch.linalg.vector_norm(y).clamp_min(R.EPS)).cpu())
    a, b = set(torch.topk(x,20,-1).indices[0].tolist()), set(torch.topk(y,20,-1).indices[0].tolist())
    return {"future_kl": kl, "logit_relative_l2": rel, "top1_match": int(x.argmax(-1).item()==y.argmax(-1).item()), "top20_overlap": len(a&b)/20, "nonfinite": int((~torch.isfinite(x)).sum().cpu())}


def prefill(F, model, tokenizer_ids, probe, basis, layers, device, name):
    prefix = torch.tensor([tokenizer_ids[:128]], dtype=torch.long, device=device); mask = torch.ones_like(prefix)
    probe.begin(F.H.plan(name, basis, False), 0)
    with torch.inference_mode():
        output = model(input_ids=prefix, attention_mask=mask, cache_position=torch.arange(128, device=device), use_cache=True)
    probe.end()
    return F.H.branch(name, basis, False, output.past_key_values, mask)


def trajectory(F, model, ids, probe, layers, device, method, mats, h, maximum, references=None):
    basis = "rotated" if rotated(method) else "native"
    item = prefill(F, model, ids, probe, basis, layers, device, method)
    if references is not None: quantize_cache(F, item["past"], layers)
    logits, states, rows = {}, {}, []
    identity = torch.eye(128, dtype=torch.float32)
    for step in range(1, maximum+1):
        value, _records = F.H.advance(model, probe, item, int(ids[127+step]), 127+step, layers, identity, step)
        if references is not None: quantize_cache(F, item["past"], layers)
        if step in HORIZONS or step == maximum:
            last = value[:, -1].detach().float().cpu()
            if references is None:
                logits[step] = last; states[step] = snapshot(F, item["past"], layers)
            else:
                metric = logit_metrics(last, references[0][step])
                metric["state_relative_l2"] = state_rel(recovered_snapshot(F,item["past"],layers,method,mats,h), references[1][step], layers)
                rows.append({"horizon": step, **metric})
    return (logits, states) if references is None else rows


def auc(rows):
    rows=sorted(rows,key=lambda x:x["horizon"]); area=sum((b["horizon"]-a["horizon"])*(a["future_kl"]+b["future_kl"])/2 for a,b in zip(rows,rows[1:])); return area/(rows[-1]["horizon"]-rows[0]["horizon"])


def bootstrap(diff, had, n=10000):
    rng=random.Random(20260921); effects=[]; relative=[]
    for _ in range(n):
        ix=[rng.randrange(len(diff)) for _ in diff]; e=statistics.median(diff[i] for i in ix); b=statistics.median(had[i] for i in ix); effects.append(e); relative.append(e/max(b,R.EPS))
    effects.sort(); relative.sort(); lo,hi=int(.025*(n-1)),int(.975*(n-1)); return {"resamples":n,"median_effect":statistics.median(diff),"ci95":[effects[lo],effects[hi]],"median_relative_reduction":statistics.median(diff)/max(statistics.median(had),R.EPS)}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--legacy-repo",default="/data01/user2/repos/GDN-quantization"); p.add_argument("--max-memory-gib",type=int,default=22)
    p.add_argument("--trace-dir",required=True); p.add_argument("--corpus",required=True); p.add_argument("--state-checkpoint",required=True); p.add_argument("--functional-checkpoint",required=True); p.add_argument("--output-dir",required=True); args=p.parse_args()
    F,model,tokenizer,local_probe,layers=R.load_context(args); device=F.model_input_device(model); h=R.hadamard(device)
    sb,sm=load_bank(args.state_checkpoint,layers,device); fb,fm=load_bank(args.functional_checkpoint,layers,device); banks={"Dense_State":sb,"Dense_Functional":fb}; methods=("Native_INT8","Hadamard","Dense_State","Dense_Functional")
    traces=R.load_traces(Path(args.trace_dir),"VALIDATION"); local={m:local_method(model,local_probe,layers,traces,m,banks,h,device) for m in methods}; local_probe.close()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); R.save_json(out/"heldout_local_metrics.json",local)
    docs=R.load_corpus(Path(args.corpus),"HELDOUT"); mats={m:matrices(m,banks,layers,device) for m in methods}; horizon_rows=[]; per_doc=[]
    for di,doc in enumerate(docs):
        ids=tokenizer(doc["raw_text"],add_special_tokens=False).input_ids[:1024]
        ref_probe=PerLayerHistoryProbe.build(F,{layer:torch.eye(128,device=device) for layer in layers}); ref_probe.install(model)
        references=trajectory(F,model,ids,ref_probe,layers,device,"Native_INT8",mats["Native_INT8"],h,128,None); ref_probe.close()
        for method in methods:
            probe=PerLayerHistoryProbe.build(F,mats[method]); probe.install(model)
            rows=trajectory(F,model,ids,probe,layers,device,method,mats[method],h,128,references); probe.close()
            for row in rows: horizon_rows.append({"document_id":doc["document_id"],"method":method,**row})
            per_doc.append({"document_id":doc["document_id"],"method":method,"auc":auc(rows)})
        print(f"PERSISTENT {di+1}/{len(docs)}",flush=True)
    summary={m:{str(hz):{k:sum(r[k] for r in horizon_rows if r["method"]==m and r["horizon"]==hz)/len([r for r in horizon_rows if r["method"]==m and r["horizon"]==hz]) for k in ("future_kl","logit_relative_l2","top1_match","top20_overlap","state_relative_l2","nonfinite")} for hz in HORIZONS} for m in methods}
    aucs={m:sum(r["auc"] for r in per_doc if r["method"]==m)/len(docs) for m in methods}; best=min(("Dense_State","Dense_Functional"),key=lambda m:aucs[m]); d={(r["document_id"],r["method"]):r["auc"] for r in per_doc}; had=[d[(x["document_id"],"Hadamard")] for x in docs]; learned=[d[(x["document_id"],best)] for x in docs]; boot=bootstrap([a-b for a,b in zip(had,learned)],had)
    verdict="CLEAR_HEADROOM" if boot["median_effect"]>0 and boot["median_relative_reduction"]>=.1 and boot["ci95"][0]>0 else ("PROMISING_HEADROOM" if boot["median_effect"]>0 else "NO_CLEAR_HEADROOM")
    result={"task":R.TASK,"model":"Ling-3.0-tiny/KDA","horizons":HORIZONS,"summary":summary,"auc":aucs,"per_document_auc":per_doc,"best_learned":best,"bootstrap_hadamard_minus_best":boot,"persistent_headroom":verdict,"KDA_ROTATION_SEMANTICS_VERSION":"CORRECTED_PREFILL_ENDPOINT_V2","REDUNDANT_PREFILL_ENDPOINT_ROTATION":"NO","checkpoint_metadata":{"Dense_State":{k:sm[k] for k in ("step","lr","objective","seed")},"Dense_Functional":{k:fm[k] for k in ("step","lr","objective","seed")}},"AIME26_used":False}
    R.save_json(out/"persistent_future_kl.json",result); R.save_json(out/"persistent_horizon_rows.json",horizon_rows); print(json.dumps({"auc":aucs,"best":best,"bootstrap":boot,"verdict":verdict},indent=2),flush=True)


if __name__=="__main__": main()
