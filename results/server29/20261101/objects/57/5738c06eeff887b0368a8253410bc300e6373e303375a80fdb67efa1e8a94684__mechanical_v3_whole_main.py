def main(gpu, label, condition):
    assert gpu in [0, 1] and os.environ['CUDA_VISIBLE_DEVICES'] == str(gpu)
    assert torch.cuda.device_count() == 1 and '4090' in torch.cuda.get_device_name(0)
    assert hardware()['gpus'][gpu]['gpu_uuid'] == UUIDS[gpu]
    plan = load(ROOT / 'protocol/contemporary_determinism_plan.json')
    trace_path = TRACES / plan['input']
    assert sha(trace_path) == plan['input_sha256']
    t = trainer()
    cfg = load(ROOT / 'configs' / ('l7r.json' if condition == 'L7R' else 'l8.json'))
    args = argparse.Namespace(legacy_repo=str(LEGACY), max_memory_gib=22, seed=0)
    random.seed(0)
    torch.manual_seed(0)
    os.environ['LING_MODEL_PATH'] = str(MODEL)
    full, cayley, ortho, model, tok, training_probe, layers = t.load_context(args)
    training_probe.close()
    assert list(layers) == LAYERS and all((p.device.type == 'cuda' for p in model.parameters()))
    device = full.model_input_device(model)
    h = ortho.normalized_hadamard(128).to(device).float()
    trace = t.load_trace(trace_path)
    ids = [int(x) for x in trace['token_ids'][:4]]
    assert ids == plan['input_token_ids']
    base = base_identity(model)
    forwards = fp_h_forward(full, model, h, ids) if condition == 'L7R' else None
    adapt(t, cayley, condition)
    bank = cayley.PerLayerCayleyRotations(tuple(layers)).to(device)
    before = bank_identity(bank)
    rot = t.final_rotations(bank, h, layers)
    for l in layers:
        assert torch.equal(rot[l], h)
        active = condition == 'L7R' or l not in REGIONS['early']
        assert before[str(l)]['requires_grad'] == before[str(l)]['in_optimizer_list'] == active
    initial_rot = {str(l): safe_tensor(r) for l, r in rot.items()}
    records = {l: t.to_device_record(trace['layer_records'][l], device) for l in layers}
    calls = []
    local = []
    qdq = []
    orig_fused = t.fused
    orig_local = t.local_output
    orig_qdq = t.exact_qdq

    def observed_fused(probe, record, state, value, token):
        out, nxt = orig_fused(probe, record, state, value, token)
        calls.append(dict(call=len(calls), token=token, input_state=safe_tensor(state), value=safe_tensor(value), core_output=safe_tensor(out), next_state=safe_tensor(nxt)))
        return (out, nxt)

    def observed_local(model, layer, core, gate):
        out = orig_local(model, layer, core, gate)
        local.append(dict(layer=layer, output=safe_tensor(out)))
        return out

    def observed_qdq(value, ste):
        out = orig_qdq(cayley, value, ste)
        qdq.append(dict(ste=ste, output=safe_tensor(out.dequant)))
        return out
    t.fused = observed_fused
    t.local_output = observed_local
    t.exact_qdq = lambda c, v, s: observed_qdq(v, s)
    provenance = []
    teacher, student = t.init_states(trace, records, rot, cayley, True)
    values = t.rollout_chunk(model=model, probe=training_probe, cayley=cayley, bank=bank, h=h, trace=trace, records=records, teacher=teacher, student=student, start=0, end=4, objective='L7_RECURRENT_DENSE_FUNCTIONAL', grad=True, provenance=provenance)
    assert all((torch.isfinite(v).item() for v in values[:4]))
    values[0].backward()
    raw = {}
    for l in layers:
        grad = bank.layer(l).theta.grad
        active = condition == 'L7R' or l not in REGIONS['early']
        if active:
            assert grad is not None and torch.isfinite(grad).all().item() and (torch.count_nonzero(grad).item() > 0)
        else:
            assert grad is None
        raw[str(l)] = None if grad is None else safe_tensor(grad)
    optimizer = torch.optim.Adam(bank.parameters(), lr=0.003, weight_decay=0.0)
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group['params']}
    assert all(((id(bank.layer(l).theta) in optimizer_ids) == before[str(l)]['in_optimizer_list'] for l in layers))
    gradnorm = float(torch.nn.utils.clip_grad_norm_(bank.parameters(), 1.0))
    assert math.isfinite(gradnorm) and gradnorm > 0
    clipped = {str(l): None if bank.layer(l).theta.grad is None else safe_tensor(bank.layer(l).theta.grad) for l in layers}
    optimizer.step()
    after = bank_identity(bank)
    postrot = t.final_rotations(bank, h, layers)
    for l in layers:
        active = condition == 'L7R' or l not in REGIONS['early']
        assert (before[str(l)]['sha256'] != after[str(l)]['sha256']) == active
        if not active:
            assert torch.equal(postrot[l], h) and after[str(l)]['sha256'] == before[str(l)]['sha256']
    assert base_identity(model) == base and all((p.grad is None for p in model.parameters()))
    writeback = t.assert_provenance(provenance)
    assert writeback['REAL_RECURRENT_WRITEBACK_GATE'] == writeback['RECURRENT_STATE_PROVENANCE_GATE'] == 'PASS'
    orth = t.orthogonality(bank, h, tuple(layers))
    assert orth['status'] == 'PASS'
    optstate = {str(l): {k: safe_tensor(v) if torch.is_tensor(v) else v for k, v in optimizer.state.get(bank.layer(l).theta, {}).items()} for l in layers}
    boundary = torch.zeros((1, 128), device=device)
    boundary[0, 0] = 127.0
    boundary[0, 1:5] = torch.tensor([0.5, 1.5, -0.5, -1.5], device=device)
    q = cayley.ling_r128_qdq(boundary)
    assert torch.equal(q.dequant[0, 1:5], torch.tensor([0.0, 2.0, 0.0, -2.0], device=device))
    x = boundary.clone().requires_grad_(True)
    s = cayley.ling_r128_ste(x)
    assert torch.equal(s.dequant, q.dequant)
    s.dequant.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))
    numeric = dict(input_token_ids=ids, trace_sha256=sha(trace_path), model_base=base, H128=safe_tensor(h), flags=flags(), condition=condition, model_config_raw_sha256=sha(MODEL / 'config.json'), config_sha256=sha(ROOT / 'configs' / ('l7r.json' if condition == 'L7R' else 'l8.json')), FP_H_full_forwards=forwards, parameter_before=before, initial_rotations=initial_rot, training_fused_calls=calls, training_local_outputs=local, training_qdq=qdq, losses=[float(v.detach().cpu()) for v in values[:4]], raw_gradients=raw, clipped_gradients=clipped, global_gradient_norm=gradnorm, parameter_after=after, optimizer_state=optstate, final_rotations={str(l): safe_tensor(r) for l, r in postrot.items()}, final_teacher_state={str(l): safe_tensor(v) for l, v in teacher.items()}, final_student_state={str(l): safe_tensor(v) for l, v in student.items()}, writeback=writeback, provenance=provenance, orthogonality=orth, base_before_after_equal=True, quantizer_ties_even=True, STE_forward_exact=True, STE_gradient_identity=True)
    receipt = dict(gate='PASS', physical_gpu=gpu, gpu_uuid=UUIDS[gpu], process_label=label, pid=os.getpid(), condition=condition, disposable_optimizer_updates=1, formal_optimizer_updates=0, Fresh_B_treatment_forwards=0, AIME_generation=0, numeric=numeric, numeric_sha256=object_sha(numeric), runtime_identity_sha256=sha(ROOT / 'runtime/v3_runtime_identity.json'), source_freeze_sha256=sha(ROOT / 'runtime/phase_a_source_freeze.json'), gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated())
    emit(ROOT / f'runtime/gpu{gpu}_{label}_{condition}_smoke.json', receipt)
    print(json.dumps(dict(gate='PASS', physical_gpu=gpu, label=label, condition=condition, numeric_sha256=receipt['numeric_sha256'])), flush=True)
