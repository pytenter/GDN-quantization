"""Schema-preserving CPU checkpoint helpers; no model or CUDA import."""
import copy
import hashlib
import json
import types


def transfer(torch, obj, target, memo=None):
    if memo is None: memo = {}
    if id(obj) in memo: return memo[id(obj)]
    if torch.is_tensor(obj): result = obj.detach().to(target, copy=True)
    elif isinstance(obj, dict):
        result = {}; memo[id(obj)] = result
        result.update({k: transfer(torch, v, target, memo) for k, v in obj.items()})
    elif isinstance(obj, list):
        result = []; memo[id(obj)] = result
        result.extend(transfer(torch, v, target, memo) for v in obj)
    elif isinstance(obj, tuple): result = tuple(transfer(torch, v, target, memo) for v in obj)
    elif isinstance(obj, (type, types.FunctionType, types.MethodType, torch.device)): result = obj
    elif hasattr(obj, '__dict__'):
        result = copy.copy(obj); memo[id(obj)] = result
        for name, value in vars(obj).items(): setattr(result, name, transfer(torch, value, target, memo))
    else: result = obj
    memo[id(obj)] = result
    return result


def tensor_bytes(torch, value):
    return value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def fingerprint(torch, obj):
    import numpy as np
    h = hashlib.sha256(); memo = {}
    def visit(x):
        if torch.is_tensor(x):
            h.update(json.dumps([list(x.shape), str(x.dtype)]).encode())
            h.update(tensor_bytes(torch, x))
        elif isinstance(x, np.ndarray):
            h.update(json.dumps([list(x.shape), str(x.dtype)]).encode())
            h.update(x.tobytes(order='C'))
        elif isinstance(x, (dict, list, tuple)) or (hasattr(x, '__dict__') and not isinstance(x, type)):
            if id(x) in memo:
                h.update(f'alias:{memo[id(x)]}'.encode()); return
            memo[id(x)] = len(memo)
            h.update(f'{type(x).__module__}.{type(x).__qualname__}'.encode())
            if isinstance(x, dict):
                for key in sorted(x, key=str): h.update(str(key).encode()); visit(x[key])
            elif isinstance(x, (list, tuple)):
                for value in x: visit(value)
            else:
                for key in sorted(vars(x)): h.update(key.encode()); visit(getattr(x, key))
        elif isinstance(x, type): h.update(f'class:{x.__module__}.{x.__qualname__}'.encode())
        else: h.update(repr(x).encode())
    visit(obj)
    return h.hexdigest()


def audit_state(audit, condition):
    return {key: getattr(audit, key)[condition] for key in ('previous', 'writes', 'consumed', 'after_anchors', 'proofs')}


def restore_audit(audit, condition, saved):
    for key, value in saved.items(): getattr(audit, key)[condition] = value


def rng_snapshot(torch):
    import random
    import numpy as np
    return {'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(),
            'python_rng': random.getstate(), 'numpy_rng': np.random.get_state()}


def rng_restore(torch, value):
    import random
    import numpy as np
    torch.set_rng_state(value['torch_rng']); torch.cuda.set_rng_state(value['cuda_rng'])
    random.setstate(value['python_rng']); np.random.set_state(value['numpy_rng'])
