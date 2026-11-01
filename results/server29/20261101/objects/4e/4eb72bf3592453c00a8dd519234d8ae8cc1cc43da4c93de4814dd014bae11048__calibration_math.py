"""Independent CPU calibration math; no torch, model, CUDA, or AIME imports.

Inputs are actual HISTORICAL_PORT H128 helper coordinates.  Query/key
normalization is deliberately NOT reconstructed here: collection must capture
those operands after the historical native-dtype normalization and FP32 H.
"""
from dataclasses import dataclass
import math
import numpy as np

GROUP_SIZE = 32
KEY_DIM = 128
VALUE_DIM = 128
OMEGA_FLOOR = 1e-12
OMEGA_BOUNDS = (1.0 / 16.0, 16.0)
ROW_EPSILON = 1e-12
ROW_BOUNDS = (0.25, 4.0)
ROWBAL_GRID = ((0.25, 0.25), (0.50, 0.25), (0.50, 0.50))


def _finite(value, name):
    result = np.asarray(value)
    if not np.issubdtype(result.dtype, np.floating):
        raise ValueError(name + ' must be floating point')
    if not np.all(np.isfinite(result)):
        raise ValueError(name + ' is nonfinite')
    return result


def _positive(value, name):
    result = _finite(value, name)
    if np.any(result <= 0):
        raise ValueError(name + ' must be positive')
    return result


def transported_query(prepared_h_query, prepared_h_key, log_decay_g, beta,
                      query_already_scaled=False, actual_alpha=None):
    """Return alpha*(q-beta*k*(k dot q)), with alpha=exp(log_decay_g).

    query/key shape [..., actual_recurrent_heads,128]; g,beta shape [...,heads].
    H query is divided by sqrt128 exactly once unless captured after that scale.
    Paired recurrent heads are processed independently, including their gates.
    This independent reference evaluates the algebra in FP64 and returns FP32;
    actual_alpha optionally binds native FP32 exp(g) bytes; it is not a byte-emulation of the native GDN update's FP32 reduction order.
    """
    q = _finite(prepared_h_query, 'query').astype(np.float64)
    k = _finite(prepared_h_key, 'key').astype(np.float64)
    g = _finite(log_decay_g, 'log_decay_g').astype(np.float64)
    b = _finite(beta, 'beta').astype(np.float64)
    if q.shape != k.shape or q.ndim < 2 or q.shape[-1] != KEY_DIM:
        raise ValueError('q/key must have matching actual-head x128 shapes')
    if g.shape != q.shape[:-1] or b.shape != q.shape[:-1]:
        raise ValueError('head-specific gate/beta shapes do not match q/k')
    if np.any(g > 0) or np.any(b < 0) or np.any(b > 1):
        raise ValueError('native log decay must be <=0; beta must be in[0,1]')
    if not query_already_scaled:
        q = q / math.sqrt(KEY_DIM)
    alpha = np.exp(g) if actual_alpha is None else _finite(actual_alpha, 'actual_alpha').astype(np.float64)
    if alpha.shape != g.shape or np.any(alpha < 0) or np.any(alpha > 1):
        raise ValueError('actual native alpha must match head gate shape and[0,1]')
    result = alpha[..., None] * (q - b[..., None] * k * np.sum(k * q, axis=-1, keepdims=True))
    if not np.all(np.isfinite(result)):
        raise ValueError('nonfinite transported query')
    return result.astype(np.float32)


def stabilize_omega(raw, floor=OMEGA_FLOOR, lower=OMEGA_BOUNDS[0], upper=OMEGA_BOUNDS[1]):
    """Floor, per-Key-Group32 geometric normalize, then clamp; no re-normalize."""
    x = _finite(raw, 'raw omega').astype(np.float64)
    if x.ndim < 2 or x.shape[-1] != KEY_DIM or np.any(x < 0):
        raise ValueError('omega must have actual-head x128 nonnegative shape')
    if not (math.isfinite(floor) and floor > 0 and 0 < lower <= 1 <= upper and math.isfinite(upper)):
        raise ValueError('invalid frozen stabilization constants')
    groups = np.maximum(x, floor).reshape(x.shape[:-1] + (4, GROUP_SIZE))
    gm = np.exp(np.mean(np.log(groups), axis=-1, keepdims=True))
    pre = (groups / gm).reshape(x.shape)
    result = np.clip(pre, lower, upper).astype(np.float32)
    report = dict(raw_min=float(x.min()), raw_max=float(x.max()), floor=float(floor),
                  floor_count=int(np.count_nonzero(x < floor)), geometric_normalization='per_layer_actual_recurrent_head_KeyGroup32',
                  before_clamp_min=float(pre.min()), before_clamp_max=float(pre.max()),
                  clamp_lower=float(lower), clamp_upper=float(upper),
                  clamp_low_count=int(np.count_nonzero(pre < lower)),
                  clamp_high_count=int(np.count_nonzero(pre > upper)),
                  after_clamp_min=float(result.min()), after_clamp_max=float(result.max()),
                  quantiles_before_clamp=np.quantile(pre, [0,.01,.1,.5,.9,.99,1]).tolist(),
                  quantiles_after_clamp=np.quantile(result, [0,.01,.1,.5,.9,.99,1]).tolist(),
                  static_dtype='float32', second_normalization_after_clamp=False)
    return result, report


class OmegaAccumulator:
    """Document-aware sequential CPU FP64 sums, without shared-head copying."""
    def __init__(self, layer_ids, head_count=32):
        ids = tuple(int(i) for i in layer_ids)
        if not ids or len(set(ids)) != len(ids) or head_count != 32:
            raise ValueError('need unique actual layers and32 recurrent heads')
        self.sums = {i:np.zeros((head_count, KEY_DIM), np.float64) for i in ids}
        self.counts = {i:0 for i in ids}
        self.seen = set()
        self.documents = set()

    def add(self, layer_id, document_id, token_index, effective_transport, split='CAL'):
        if split != 'CAL':
            raise ValueError('VAL cannot update omega')
        if not document_id or not isinstance(token_index, int) or token_index < 1:
            raise ValueError('transport position must be actual next-token index >=1')
        if layer_id not in self.sums:
            raise ValueError('unknown layer')
        identity = (str(document_id), int(token_index), int(layer_id))
        if identity in self.seen:
            raise ValueError('duplicate document/token/layer')
        value = _finite(effective_transport, 'effective transported query').astype(np.float64)
        if value.shape != self.sums[layer_id].shape:
            raise ValueError('actual-head identity shape mismatch')
        self.sums[layer_id] += value * value
        self.counts[layer_id] += 1
        self.seen.add(identity)
        self.documents.add(str(document_id))

    def finalize(self, expected_count=None):
        if any(n == 0 for n in self.counts.values()) or len(set(self.counts.values())) != 1:
            raise ValueError('incomplete per-layer omega coverage')
        if expected_count is not None and set(self.counts.values()) != {int(expected_count)}:
            raise ValueError('unexpected number of real CAL transitions')
        values, diagnostics = {}, {}
        for layer, total in self.sums.items():
            values[layer], diagnostics[layer] = stabilize_omega(total / self.counts[layer])
        return values, dict(counts=dict(self.counts), documents=sorted(self.documents),
                            accumulation_dtype='float64', output_dtype='float32',
                            per_head_gate_beta_independent=True, layers=diagnostics)


def row_factors(state, omega, a, b, epsilon=ROW_EPSILON, lower=ROW_BOUNDS[0], upper=ROW_BOUNDS[1]):
    """Independent mathematical CPU reference; state[...,Key128,Value128]."""
    x = _finite(state, 'state').astype(np.float64)
    w = _positive(omega, 'omega').astype(np.float64)
    if x.ndim < 2 or x.shape[-2:] != (KEY_DIM, VALUE_DIM) or w.shape != x.shape[:-1]:
        raise ValueError('state/omega must preserve layer/head/key identity')
    if not all(math.isfinite(z) for z in (a,b,epsilon,lower,upper)) or epsilon <= 0 or lower <= 0 or upper < lower:
        raise ValueError('invalid frozen RowBal constants')
    m = np.sqrt(np.mean(x * x, axis=-1) + epsilon)
    grouped_m = m.reshape(m.shape[:-1] + (4,GROUP_SIZE))
    grouped_w = w.reshape(w.shape[:-1] + (4,GROUP_SIZE))
    log_m = np.log(grouped_m)
    log_w = np.log(grouped_w)
    log_r = a*(log_m - log_m.mean(-1, keepdims=True)) - b*(log_w - log_w.mean(-1, keepdims=True))
    # Clip in log space so arbitrarily large finite inputs cannot overflow exp.
    result = np.exp(np.clip(log_r, math.log(lower), math.log(upper))).reshape(w.shape).astype(np.float32)
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise ValueError('RowBal produced illegal factors')
    return result


@dataclass(frozen=True)
class CalibrationSample:
    document_id: str
    raw_document_sha256: str
    split: str
    layer_id: int
    token_index: int
    next_token_index: int
    state_h: object
    transported_next: object


def _audit_samples(cal_samples, val_samples, omega):
    if not cal_samples or not val_samples:
        raise ValueError('both document-isolated CAL and VAL samples required')
    cal_doc = {s.document_id for s in cal_samples}
    val_doc = {s.document_id for s in val_samples}
    cal_raw = {s.raw_document_sha256 for s in cal_samples}
    val_raw = {s.raw_document_sha256 for s in val_samples}
    if cal_doc & val_doc or cal_raw & val_raw:
        raise ValueError('CAL/VAL document overlap')
    seen=set()
    for expected, samples in (('CAL',cal_samples),('VAL',val_samples)):
        for s in samples:
            if s.split != expected or not s.document_id or len(s.raw_document_sha256) != 64:
                raise ValueError('sample split/document identity differs')
            if s.layer_id not in omega or s.next_token_index != s.token_index+1 or s.token_index < 0:
                raise ValueError('next-step alignment/layer mismatch')
            ident=(s.document_id,s.token_index,s.layer_id)
            if ident in seen:raise ValueError('duplicate sample')
            seen.add(ident)
            x=_finite(s.state_h,'sample state');g=_finite(s.transported_next,'sample transport')
            if x.shape != (32,128,128) or g.shape != (32,128):
                raise ValueError('sample must retain all32 actual recurrent heads')
    return dict(CAL_documents=sorted(cal_doc), VAL_documents=sorted(val_doc),
                CAL_samples=len(cal_samples), VAL_samples=len(val_samples),
                document_overlap=0, raw_document_overlap=0)


def readout_proxy_mse(state, reconstructed, effective_transport):
    x=_finite(state,'state').astype(np.float64)
    d=_finite(reconstructed,'reconstructed').astype(np.float64)
    g=_finite(effective_transport,'transport').astype(np.float64)
    if x.shape != d.shape or g.shape != x.shape[:-1]:
        raise ValueError('readout error shape mismatch')
    errors=np.sum((d-x)*g[...,None],axis=-2)
    return float(np.mean(errors*errors))


def select_rowbal_parameters(cal_samples, val_samples, omega, qdq, grid=ROWBAL_GRID):
    """Select ONLY CAL's unweighted one-step readout proxy; report fixed VAL.

    qdq(state,omega,a,b)->reconstructed must implement RowBal+canonical Sym32.
    Its callback must not execute a model or inspect AIME data.  This function
    deliberately never calls a combined WScale quantizer during selection, so
    RowBal and RW have exactly one shared (a,b).
    """
    cal_samples=tuple(cal_samples);val_samples=tuple(val_samples)
    grid=tuple(tuple(float(z) for z in p) for p in grid)
    if grid != ROWBAL_GRID:
        raise ValueError('the pre-frozen finite grid cannot be changed')
    audit=_audit_samples(cal_samples,val_samples,omega)
    rows=[]
    for index,(a,b) in enumerate(grid):
        def evaluate(samples):
            scores=[]
            for s in samples:
                decoded=qdq(s.state_h,omega[s.layer_id],a,b)
                scores.append(readout_proxy_mse(s.state_h,decoded,s.transported_next))
            return float(math.fsum(scores)/len(scores))
        rows.append(dict(grid_index=index,a=a,b=b,CAL_readout_proxy_mse=evaluate(cal_samples),
                         VAL_readout_proxy_mse=evaluate(val_samples)))
    chosen=min(rows,key=lambda r:(r['CAL_readout_proxy_mse'],r['grid_index']))
    return dict(status='PASS_CAL_ONLY_FINITE_GRID_SELECTION',selected_a=chosen['a'],selected_b=chosen['b'],
                selected_grid_index=chosen['grid_index'],selection_proxy='mean((E_t.T @ A_next.T @ q_next)^2)',
                tie_break='first in pre-frozen grid order for exactly equal CAL proxy',
                selected_on='CAL_only',VAL_used_for_selection=False,rows=rows,
                shared_by=('rowbal','rowbal_wscale'),omega_shared_by=('wscale','rowbal','rowbal_wscale'),
                AIME_used=False,**audit)
