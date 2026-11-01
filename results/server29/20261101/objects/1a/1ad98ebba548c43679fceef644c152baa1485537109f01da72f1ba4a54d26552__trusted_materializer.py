"""Explicit-storage materialization; exact full comparisons, no repair path."""
import hashlib
import time
import numpy as np
import torch

CHUNK = 1 << 20


class IntegrityFailure(RuntimeError):
    def __init__(self, name, result):
        super().__init__(f"exact model integrity failed: {name}")
        self.name, self.result = name, result


def fingerprint(tensor, expected=None):
    start = time.monotonic()
    if tensor.is_meta:
        raise RuntimeError('meta state cannot be verified')
    # Canonical checkpoint weights are contiguous; do not allocate a GPU clone.
    if not tensor.is_contiguous():
        raise RuntimeError('noncontiguous state needs an explicitly audited chunk layout')
    flat = tensor.detach().view(-1)
    reference = expected.detach().view(-1) if expected is not None else None
    if reference is not None and (tensor.shape != expected.shape or tensor.dtype != expected.dtype):
        raise RuntimeError('shape/dtype differs from intended CPU source')
    digest = hashlib.sha256()
    finite_count = nan_count = inf_count = mismatch = 0
    absmax = 0.0
    first = []
    for offset in range(0, flat.numel(), CHUNK):
        cpu = flat[offset:offset + CHUNK].cpu().contiguous()
        raw = cpu.view(torch.uint8).numpy()
        digest.update(raw.tobytes())
        if cpu.dtype == torch.bfloat16:
            values = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        else:
            values = cpu.numpy()
        good = np.isfinite(values)
        finite_count += int(good.sum())
        nan_count += int(np.isnan(values).sum())
        inf_count += int(np.isinf(values).sum())
        if good.any():
            absmax = max(absmax, float(np.abs(values[good]).max()))
        if reference is not None:
            eraw = reference[offset:offset + CHUNK].contiguous().view(torch.uint8).numpy()
            bad = np.flatnonzero(raw != eraw)
            mismatch += int(bad.size)
            for index in bad[:max(0, 16 - len(first))]:
                first.append({'byte_offset': offset * tensor.element_size() + int(index),
                              'actual': int(raw[index]), 'expected': int(eraw[index])})
    return {'shape': list(tensor.shape), 'stride': list(tensor.stride()),
            'dtype': str(tensor.dtype), 'numel': tensor.numel(),
            'bytes': tensor.numel() * tensor.element_size(), 'raw_sha256': digest.hexdigest(),
            'finite': finite_count == tensor.numel(), 'finite_count': finite_count,
            'nan_count': nan_count, 'inf_count': inf_count, 'absmax': absmax,
            'exact_equal': None if expected is None else mismatch == 0,
            'mismatch_bytes': mismatch, 'first_mismatched_bytes': first,
            'verification_seconds': time.monotonic() - start}


def state_items(model):
    result = {}
    for prefix, module in model.named_modules(remove_duplicate=False):
        for kind, items in [('PARAMETER', module._parameters), ('BUFFER', module._buffers)]:
            for local_name, tensor in items.items():
                if tensor is not None:
                    name = f'{prefix}.{local_name}' if prefix else local_name
                    if name in result:
                        raise RuntimeError(f'duplicate runtime path: {name}')
                    result[name] = {'tensor': tensor, 'module': module, 'local_name': local_name,
                                    'kind': kind, 'persistent': kind == 'PARAMETER' or
                                    local_name not in module._non_persistent_buffers_set}
    return result


def alias_structure(items):
    objects, storages = {}, {}
    for name, item in items.items():
        t = item['tensor']
        if t.is_meta:
            raise RuntimeError(f'unmaterialized CPU state: {name}')
        objects.setdefault((item['kind'], id(t)), []).append(name)
        storages.setdefault(t.untyped_storage()._cdata, []).append(name)
    return {'object_groups': sorted(sorted(v) for v in objects.values()),
            'storage_groups': sorted(sorted(v) for v in storages.values()),
            'layouts': {n: {'shape': list(x['tensor'].shape), 'stride': list(x['tensor'].stride()),
                             'offset': x['tensor'].storage_offset(), 'dtype': str(x['tensor'].dtype),
                             'kind': x['kind'], 'requires_grad': x['tensor'].requires_grad}
                        for n, x in sorted(items.items())}}


def materialize(model, emit, device='cuda:0'):
    items = state_items(model)
    baseline = alias_structure(items)
    # Keep verified CPU references throughout installation and all later guards.
    sources = {name: x['tensor'].detach() for name, x in items.items()}
    groups = {}
    for name, item in items.items():
        groups.setdefault(item['tensor'].untyped_storage()._cdata, []).append(name)
    new_objects = {}
    for names in groups.values():
        first = items[names[0]]['tensor']
        if first.device.type != 'cpu':
            raise RuntimeError('materializer expects canonical CPU state only')
        if any(items[n]['tensor'].dtype != first.dtype for n in names):
            raise RuntimeError('mixed-dtype aliased storage requires explicit review')
        elements = first.untyped_storage().nbytes() // first.element_size()
        source_storage = first.detach().as_strided((elements,), (1,), storage_offset=0)
        destination = torch.empty((elements,), device=device, dtype=first.dtype)
        with torch.no_grad():
            destination.copy_(source_storage, non_blocking=False)
        if str(device).startswith('cuda'):
            torch.cuda.synchronize(device)
        candidates = {}
        # Verify all views BEFORE binding any view in this storage group.
        for name in names:
            old = items[name]['tensor']
            view = destination.as_strided(old.shape, old.stride(), old.storage_offset())
            result = fingerprint(view, sources[name])
            emit(name, items[name]['kind'], 'IMMEDIATE_COPY', result)
            if not result['exact_equal'] or not result['finite']:
                raise IntegrityFailure(name, result)
            candidates[name] = view
        for name in names:
            item = items[name]
            old = item['tensor']
            identity = (item['kind'], id(old))
            if identity not in new_objects:
                view = candidates[name]
                new_objects[identity] = (torch.nn.Parameter(view, requires_grad=old.requires_grad)
                                         if item['kind'] == 'PARAMETER' else view)
            if item['kind'] == 'PARAMETER':
                item['module']._parameters[item['local_name']] = new_objects[identity]
            else:
                item['module']._buffers[item['local_name']] = new_objects[identity]
        del destination, candidates
    actual_aliases = alias_structure(state_items(model))
    if actual_aliases != baseline:
        raise RuntimeError('object/storage/layout alias structure changed')
    return sources, baseline


def verify_model(model, sources, emit, stage):
    items = state_items(model)
    if set(items) != set(sources):
        raise RuntimeError('immutable runtime state inventory changed')
    hashes = {}
    for name, item in items.items():
        result = fingerprint(item['tensor'], sources[name])
        emit(name, item['kind'], stage, result)
        if not result['exact_equal'] or not result['finite']:
            raise IntegrityFailure(name, result)
        hashes[name] = result['raw_sha256']
    return hashes
