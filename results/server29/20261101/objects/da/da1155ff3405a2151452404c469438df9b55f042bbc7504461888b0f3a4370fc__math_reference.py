"""Independent CPU-only scalar FP32 reference; no torch, model, or GPU imports.

This deliberately slow reference is for numerical qualification, never the
production per-token implementation. State layout is [B,H,Key128,Value128].
It neither runs a model nor owns a persistent unquantized continuation state.
"""
from array import array
from dataclasses import dataclass
import math
import struct

PACK = struct.Struct('<f')
KEY = VALUE = 128
GROUP = 32
PEAK_FLOOR = 1e-12
OMEGA_FLOOR = 1e-12
ROW_RMS_EPS = 1e-12
LAMBDAS = (0.60, 0.75, 0.875, 1.00, 1.125)
TIE_ORDER = (3, 2, 4, 1, 0)  # nearest 1, then numerical lambda ascending
ROW_PARAMETER_GRID = ((0.25, 0.25), (0.50, 0.25), (0.50, 0.50))
METHODS = ('base', 'wscale', 'rowbal', 'rowbal_wscale')


def f32(x):
    return PACK.unpack(PACK.pack(float(x)))[0]


def add(x, y): return f32(f32(x) + f32(y))
def mul(x, y): return f32(f32(x) * f32(y))
def div(x, y): return f32(f32(x) / f32(y))


def finite(values, label):
    if any(not math.isfinite(float(v)) for v in values):
        raise FloatingPointError('nonfinite ' + label)


def tree_sum(values):
    """Fixed halves-first binary FP32 reduction, with power-of-two zero pad."""
    v = [f32(x) for x in values]
    if not v: return 0.0
    n = 1 << (len(v)-1).bit_length()
    v += [0.0] * (n-len(v))
    while n > 1:
        half = n // 2
        v = [add(v[i], v[i+half]) for i in range(half)]
        n = half
    return v[0]


@dataclass
class Tensor32:
    shape: tuple
    values: array

    def __init__(self, shape, values):
        self.shape = tuple(shape)
        if any(type(d) is not int or d <= 0 for d in self.shape):
            raise ValueError('positive integer dimensions required')
        self.values = array('f', values)
        if math.prod(self.shape) != len(self.values):
            raise ValueError('tensor size mismatch')

    def bytes(self):
        return b''.join(PACK.pack(x) for x in self.values)


@dataclass
class Encoded:
    method: str
    shape: tuple
    codes: array  # signed integer [-127,127], in original state order
    scales: Tensor32  # [B,H,4,128]
    negative_zero: bytes
    row_factors: Tensor32  # [B,H,128], exactly 1 for base/WScale
    candidate_indices: bytes  # [B,H,4,128], indices in LAMBDAS
    losses: Tensor32  # selected original-X-space weighted loss
    arithmetic: str


def canonical_scale(peak, arithmetic='CUDA_LEGACY_EMULATED'):
    safe = max(f32(peak), f32(PEAK_FLOOR))
    if arithmetic == 'CUDA_LEGACY_EMULATED':
        return mul(safe, f32(1.0/127.0))
    if arithmetic == 'CPU_CANONICAL':
        return div(safe, 127.0)
    raise ValueError('unknown original arithmetic graph')


def candidate_scale(s0, candidate, arithmetic='CUDA_LEGACY_EMULATED'):
    # lambda=1 retains the exact inherited absmax candidate, including floor.
    return max(mul(s0, f32(candidate)), canonical_scale(0.0, arithmetic))


def sym8_scalar(y, scale):
    """Canonical nearest-even rounding, saturation, and signed-zero decode."""
    finite((y, scale), 'scalar input/scale')
    if scale <= 0: raise FloatingPointError('nonpositive scale')
    level = div(y, scale)
    if not math.isfinite(level): raise FloatingPointError('nonfinite level')
    code = max(-127, min(127, round(level)))
    negative_zero = code == 0 and math.copysign(1.0, level) < 0
    reconstructed = mul(float(code), scale)
    if negative_zero: reconstructed = -0.0
    return int(code), negative_zero, reconstructed


def choose_candidate(losses):
    """Exact finite FP32 loss comparison; no fuzzy/posthoc tie threshold."""
    if len(losses) != len(LAMBDAS): raise ValueError('five losses required')
    finite(losses, 'candidate loss')
    return min(TIE_ORDER, key=lambda index: losses[index])


def _gm32(values):
    if len(values) != GROUP or any(v <= 0 for v in values):
        raise ValueError('32 positive values required for geometric mean')
    finite(values, 'geometric mean input')
    logs = [f32(math.log(f32(v))) for v in values]
    return f32(math.exp(mul(tree_sum(logs), 1.0/GROUP)))


def stabilize_omega(raw, lower=1/16, upper=16.0):
    """Per-head/per-G32 raw squared transported-query mean stabilization.

    FP32 raw floor -> FP32 log-GM -> FP32 division -> clamp. No renormalize
    after clamp. Returns the one shared static omega and distribution audit.
    """
    if not isinstance(raw, Tensor32) or len(raw.shape) != 2 or raw.shape[-1] != KEY:
        raise ValueError('raw omega must have shape [actual_head,128]')
    finite(raw.values, 'raw omega')
    if any(v < 0 for v in raw.values): raise ValueError('negative squared sensitivity')
    out=[]; normalized=[]; nlow=nhigh=0
    for start in range(0, len(raw.values), GROUP):
        safe=[max(f32(OMEGA_FLOOR), v) for v in raw.values[start:start+GROUP]]
        gm=_gm32(safe)
        for v in safe:
            n=div(v, gm); normalized.append(n)
            nlow += n < lower; nhigh += n > upper
            out.append(f32(min(upper, max(lower, n))))
    return Tensor32(raw.shape,out), dict(raw_min=min(raw.values),raw_max=max(raw.values),
        normalized_min=min(normalized),normalized_max=max(normalized),
        clamped_min=min(out),clamped_max=max(out),clamp_low_count=nlow,
        clamp_high_count=nhigh,raw_floor=f32(OMEGA_FLOOR),lower=lower,upper=upper)


def row_factors(state, omega, a, b):
    _validate(state, omega)
    if (a,b) not in ROW_PARAMETER_GRID and (a,b) != (0.0,0.0):
        raise ValueError('parameters outside frozen CAL selection grid')
    batch,heads,_,_=state.shape
    out=[]
    for bh in range(batch*heads):
        head=bh%heads
        rms=[]
        for row in range(KEY):
            start=(bh*KEY+row)*VALUE
            squares=[mul(v,v) for v in state.values[start:start+VALUE]]
            m=f32(math.sqrt(add(mul(tree_sum(squares),1.0/VALUE),ROW_RMS_EPS)))
            rms.append(m)
        for group in range(4):
            row0=group*GROUP
            local_rms=rms[row0:row0+GROUP]
            weights=list(omega.values[head*KEY+row0:head*KEY+row0+GROUP])
            gm_m=_gm32(local_rms); gm_w=_gm32(weights)
            for m,w in zip(local_rms,weights):
                # Each pow and product is individually rounded FP32.
                amplitude=f32(div(m,gm_m)**f32(a))
                sensitivity=f32(div(w,gm_w)**f32(-b))
                r=mul(amplitude,sensitivity)
                out.append(f32(min(4.0,max(0.25,r))))
    finite(out,'row factor')
    return Tensor32((batch,heads,KEY),out)


def _validate(state, omega):
    if not isinstance(state, Tensor32) or len(state.shape)!=4 or state.shape[-2:]!=(KEY,VALUE):
        raise ValueError('expected FP32 [B,H,128,128]')
    finite(state.values,'pre-QDQ state')
    if not isinstance(omega,Tensor32) or omega.shape!=(state.shape[1],KEY):
        raise ValueError('omega must bind every actual recurrent head [H,128]')
    finite(omega.values,'omega')
    if any(w<=0 for w in omega.values): raise ValueError('omega must be strictly positive')


def quantize(state, omega, method, *, a=None, b=None,
             arithmetic='CUDA_LEGACY_EMULATED', force_unit_row=False,
             force_lambda_one=False, row_factors_override=None):
    _validate(state,omega)
    if method not in METHODS: raise ValueError('unknown quantizer method')
    batch,heads,_,_=state.shape
    use_row=method in ('rowbal','rowbal_wscale') and not force_unit_row
    use_search=method in ('wscale','rowbal_wscale') and not force_lambda_one
    if row_factors_override is not None:
        if not use_row or row_factors_override.shape!=(batch,heads,KEY):
            raise ValueError('qualification-only factor override shape/method mismatch')
        finite(row_factors_override.values,'qualification row factors')
        if any(not .25<=r<=4 for r in row_factors_override.values):
            raise ValueError('qualification factor outside frozen clamp')
        factors=row_factors_override
    elif use_row:
        if a is None or b is None: raise ValueError('selected frozen (a,b) required')
        factors=row_factors(state,omega,a,b)
    else: factors=Tensor32((batch,heads,KEY),[1.0]*(batch*heads*KEY))
    codes=array('b',[0])*len(state.values); signs=bytearray(len(codes))
    scales=[]; choices=bytearray(); selected_losses=[]
    for bh in range(batch*heads):
        head=bh%heads
        for group in range(4):
            row0=GROUP*group
            for column in range(VALUE):
                offsets=[(bh*KEY+row0+i)*VALUE+column for i in range(GROUP)]
                x=[state.values[o] for o in offsets]
                r=list(factors.values[bh*KEY+row0:bh*KEY+row0+GROUP])
                w=list(omega.values[head*KEY+row0:head*KEY+row0+GROUP])
                y=[div(v,rf) for v,rf in zip(x,r)]
                s0=canonical_scale(max(abs(v) for v in y),arithmetic)
                candidates=range(len(LAMBDAS)) if use_search else (3,)
                trials={}; losses=[math.inf]*len(LAMBDAS)
                for candidate in candidates:
                    scale=candidate_scale(s0,LAMBDAS[candidate],arithmetic)
                    fields=[sym8_scalar(v,scale) for v in y]
                    # Canonical Sym8 decode first, then positive row restore.
                    decoded=[mul(field[2],rf) for field,rf in zip(fields,r)]
                    diffs=[add(v,-d) for v,d in zip(x,decoded)]
                    loss=tree_sum([mul(weight,mul(d,d)) for weight,d in zip(w,diffs)])
                    if not math.isfinite(loss): raise FloatingPointError('nonfinite weighted loss')
                    losses[candidate]=loss; trials[candidate]=(scale,fields)
                choice=choose_candidate(losses) if use_search else 3
                scale,fields=trials[choice]
                scales.append(scale); choices.append(choice); selected_losses.append(losses[choice])
                for offset,field in zip(offsets,fields):
                    codes[offset]=field[0]; signs[offset]=field[1]
    return Encoded(method,state.shape,codes,Tensor32((batch,heads,4,VALUE),scales),
                   bytes(signs),factors,bytes(choices),
                   Tensor32((batch,heads,4,VALUE),selected_losses),arithmetic)


def decode(encoded):
    batch,heads,_,_=encoded.shape
    out=[]
    for bh in range(batch*heads):
        for row in range(KEY):
            rf=encoded.row_factors.values[bh*KEY+row]
            for column in range(VALUE):
                offset=(bh*KEY+row)*VALUE+column
                scale=encoded.scales.values[(bh*4+row//GROUP)*VALUE+column]
                d=mul(float(encoded.codes[offset]),scale)
                if encoded.negative_zero[offset]: d=-0.0
                out.append(mul(d,rf))
    finite(out,'decode')
    return Tensor32(encoded.shape,out)


def qdq_writeback(state, omega, method, **kwargs):
    """Immediate overwrite of caller's FP32 carrier; no continuation copy."""
    encoded=quantize(state,omega,method,**kwargs)
    state.values[:]=decode(encoded).values
    return encoded


def transported_query(query, key, log_gate, beta, *, query_already_scaled=False):
    """Actual per-recurrent-head H-coordinate helper q/k -> A.T @ readout q.

    Inputs q/k are post canonical BF16 normalization + FP32 Hadamard transform,
    prior to the native helper's 1/sqrt(128) query factor unless explicitly set.
    beta is the actual sigmoid(b), log_gate the actual negative gate g. This
    function never infers/duplicates these fields from shared 16-head q/k.
    """
    if len(query)!=KEY or len(key)!=KEY: raise ValueError('Key128 vectors required')
    finite((*query,*key,log_gate,beta),'transport inputs')
    if not 0<=beta<=1 or log_gate>0: raise ValueError('invalid native gate/beta')
    q=[f32(v) if query_already_scaled else mul(v,f32(1/math.sqrt(KEY))) for v in query]
    k=[f32(v) for v in key]
    alpha=f32(math.exp(f32(log_gate)))
    dot=tree_sum([mul(ki,qi) for ki,qi in zip(k,q)])
    return [mul(alpha,add(qi,-mul(mul(beta,ki),dot))) for ki,qi in zip(k,q)]
