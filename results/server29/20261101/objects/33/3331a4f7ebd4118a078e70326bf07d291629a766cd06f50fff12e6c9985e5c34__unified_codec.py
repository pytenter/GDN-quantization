"""Unified Sym8 G32 WScale/RowBal codec; CUDA/Triton imports are lazy.

No model or device initialization happens on import. Kernels operate on the
current post-readout state only. Encoded.cpu() never retains an FP32 state.
"""
from dataclasses import dataclass

METHODS={'base':'base','W':'wscale','R':'rowbal','RW':'rowbal_wscale',
         'wscale':'wscale','rowbal':'rowbal','rowbal_wscale':'rowbal_wscale'}
LAMBDAS=(.60,.75,.875,1.,1.125)
PARAMETER_GRID=((.25,.25),(.50,.25),(.50,.50))
PEAK_FLOOR=1e-12
ROW_EPS=1e-12
_KERNELS=None


@dataclass
class Encoded:
    method: str
    shape: tuple
    codes: object
    scales: object
    zero_points: object
    negative_zero_flags: object
    row_factors: object
    lambda_indices: object
    weighted_losses: object = None
    invalid_flags: object = None
    _decoded: object = None

    def cpu(self):
        def c(t):return None if t is None else t.detach().cpu().clone()
        return Encoded(self.method,self.shape,c(self.codes),c(self.scales),None,
            c(self.negative_zero_flags),c(self.row_factors),c(self.lambda_indices),
            c(self.weighted_losses),c(self.invalid_flags),None)


def tensor_bytes(t):
    return t.detach().cpu().contiguous().view(__import__('torch').uint8).numpy().tobytes()


def slice_encoded(e,index):
    if type(index) is not int or not 0<=index<e.shape[0]:raise IndexError(index)
    def s(t):return None if t is None else t[index:index+1]
    return Encoded(e.method,(1,*e.shape[1:]),s(e.codes),s(e.scales),None,
        s(e.negative_zero_flags),s(e.row_factors),s(e.lambda_indices),
        s(e.weighted_losses),e.invalid_flags,s(e._decoded))


def decode(e):
    import torch
    if e.codes.dtype!=torch.int8:raise ValueError('symmetric signed INT8 required')
    if e._decoded is not None:return e._decoded
    out=e.codes.float()*e.scales.unsqueeze(-2)
    out=torch.where(e.negative_zero_flags,torch.full_like(out,-0.0),out)
    out=out*e.row_factors.reshape(*e.shape[:2],4,32,1)
    return out.reshape(e.shape)


def require_consumed(actual,expected_encoded):
    import torch
    expected=decode(expected_encoded)
    actual_cpu=actual.detach().cpu()
    if actual_cpu.shape!=expected.shape or actual_cpu.dtype!=expected.dtype:
        raise RuntimeError('next-consumption shape/dtype mismatch')
    if not torch.equal(actual_cpu,expected) or tensor_bytes(actual_cpu)!=tensor_bytes(expected):
        raise RuntimeError('next helper did not consume actual QDQ bytes')
    return True


def validate_encoded(e):
    if e.invalid_flags is None or int(e.invalid_flags.item())!=0:
        raise FloatingPointError('nonfinite/illegal quantizer field; no state writeback allowed')
    return True


# These undecorated functions are compiled lazily after tl/libdevice globals
# have been installed. This keeps pre-GPU source authorization CPU-only.
def _halves_add_first(X,REDUCTION,WIDTH):
    # Logical [N,W] -> [2,N/2,W] -> [N/2,W,2]. split selects the
    # two complete halves before the FP32 addition; it is not an adjacent
    # pair reduction. This gives Y[i,j] = FP32(X[i,j]+X[i+N/2,j]).
    halves=tl.permute(tl.reshape(X,(2,REDUCTION//2,WIDTH)),(1,2,0))
    low,high=tl.split(halves)
    return low+high


def _halves_sum_first(X,REDUCTION,WIDTH):
    # Every condition is constexpr. Explicit dependencies preserve the
    # preregistered halves-first FP32 graph regardless of warp/CTA layout.
    # No generic reduction, lane mask, or posthoc tolerance defines this sum.
    tl.static_assert((REDUCTION==32) or (REDUCTION==128),
                     'Only frozen Key32/Value128 reductions are admitted')
    value=X
    if REDUCTION==128:
        value=_halves_add_first(value,128,WIDTH)
        value=_halves_add_first(value,64,WIDTH)
    value=_halves_add_first(value,32,WIDTH)
    value=_halves_add_first(value,16,WIDTH)
    value=_halves_add_first(value,8,WIDTH)
    value=_halves_add_first(value,4,WIDTH)
    value=_halves_add_first(value,2,WIDTH)
    return tl.reshape(value,(WIDTH,))


def _row_kernel(X,W,R,ERROR,HEADS,OMEGA_BATCHED,A,B):
    pid=tl.program_id(0)
    bh=pid//4; group=pid%4
    rows=group*32+tl.arange(0,32)
    columns=tl.arange(0,128)
    x=tl.load(X+bh*16384+rows[:,None]*128+columns[None,:])
    omega_base=bh*128 if OMEGA_BATCHED else (bh%HEADS)*128
    omega=tl.load(W+omega_base+rows)
    mean_sq=_halves_sum_first(tl.permute(x*x,(1,0)),128,32)*0.0078125
    rms=tl.sqrt(mean_sq+1e-12)
    sum_log_m=_halves_sum_first(tl.reshape(tl.log(rms),(32,1)),32,1)
    sum_log_w=_halves_sum_first(tl.reshape(tl.log(omega),(32,1)),32,1)
    # Triton3.1 forbids reshaping a block to rank0. Keep the single value
    # as [1]; broadcasting to the32 rows changes no FP32 arithmetic node.
    gm_m=tl.exp(sum_log_m*0.03125)
    gm_w=tl.exp(sum_log_w*0.03125)
    ratio_m=tl.div_rn(rms,gm_m)
    ratio_w=tl.div_rn(omega,gm_w)
    # Grid exponents are dyadic. pow is log/exp in this frozen kernel.
    amplitude=tl.exp(tl.log(ratio_m)*A)
    sensitivity=tl.exp(tl.log(ratio_w)*(-B))
    raw=amplitude*sensitivity
    factor=tl.minimum(4.,tl.maximum(.25,raw))
    valid_x=tl.sum((tl.abs(x)<float('inf')).to(tl.int32),axis=1)==128
    valid=(valid_x & (omega>0) & (tl.abs(omega)<float('inf')) &
        (rms>0) & (tl.abs(rms)<float('inf')) &
        (gm_m>0) & (tl.abs(gm_m)<float('inf')) &
        (gm_w>0) & (tl.abs(gm_w)<float('inf')) &
        (raw>0) & (tl.abs(raw)<float('inf')) &
        (factor>0) & (tl.abs(factor)<float('inf')))
    if tl.sum((~valid).to(tl.int32),axis=0)>0:tl.atomic_or(ERROR,1)
    tl.store(R+bh*128+rows,factor)


def _group_kernel(X,W,R,CODES,SCALES,SIGNS,CHOICES,LOSSES,DECODED,ERROR,
                  HEADS,OMEGA_BATCHED,USE_ROW,SEARCH):
    pid=tl.program_id(0)
    group_id=pid//8; block_column=pid%8
    bh=group_id//4; group=group_id%4
    row=group*32+tl.arange(0,32)
    column=block_column*16+tl.arange(0,16)
    offsets=bh*16384+row[:,None]*128+column[None,:]
    x=tl.load(X+offsets)
    r=tl.load(R+bh*128+row) if USE_ROW else tl.full((32,),1.,tl.float32)
    omega_base=bh*128 if OMEGA_BATCHED else (bh%HEADS)*128
    w=tl.load(W+omega_base+row)
    y=tl.div_rn(x,r[:,None])
    peak=tl.max(tl.abs(y),axis=0)
    safe=tl.maximum(peak,1e-12)
    s0=safe*0.0078740157186985015869140625
    floor=tl.full((16,),1e-12,tl.float32)*0.0078740157186985015869140625
    best_loss=tl.full((16,),float('inf'),tl.float32)
    best_scale=s0
    best_code=tl.full((32,16),0,tl.int32)
    best_sign=tl.full((32,16),False,tl.int1)
    best_decoded=tl.full((32,16),0.,tl.float32)
    best_index=tl.full((16,),3,tl.int32)
    any_invalid=tl.full((16,),False,tl.int1)
    for ci in tl.static_range(5 if SEARCH else 1):
        # Visit exact tie priority first; strict less-than preserves ties.
        if ci==0:lam=1.;index=3
        elif ci==1:lam=.875;index=2
        elif ci==2:lam=1.125;index=4
        elif ci==3:lam=.75;index=1
        else:lam=.60;index=0
        scale=tl.maximum(s0*lam,floor)
        level=tl.div_rn(y,scale[None,:])
        rounded=libdevice.nearbyint(level)
        clipped=tl.minimum(127.,tl.maximum(-127.,rounded))
        code=clipped.to(tl.int32)
        negative_zero=(clipped==0) & (clipped.to(tl.int32,bitcast=True)<0)
        d=code.to(tl.float32)*scale[None,:]
        signed_negative_zero=tl.full((32,16),-2147483648,tl.int32).to(tl.float32,bitcast=True)
        d=tl.where(negative_zero,signed_negative_zero,d)
        d=d*r[:,None]
        difference=x-d
        square=difference*difference
        term=w[:,None]*square
        loss=_halves_sum_first(term,32,16)
        valid_level=tl.sum((tl.abs(level)<float('inf')).to(tl.int32),axis=0)==32
        valid_decoded=tl.sum((tl.abs(d)<float('inf')).to(tl.int32),axis=0)==32
        any_invalid=any_invalid | (~valid_level) | (~valid_decoded) | \
            (~(scale>0)) | (~(tl.abs(scale)<float('inf'))) | (~(tl.abs(loss)<float('inf')))
        better=loss<best_loss
        best_loss=tl.where(better,loss,best_loss)
        best_scale=tl.where(better,scale,best_scale)
        best_index=tl.where(better,index,best_index)
        best_code=tl.where(better[None,:],code,best_code)
        best_sign=tl.where(better[None,:],negative_zero,best_sign)
        best_decoded=tl.where(better[None,:],d,best_decoded)
    valid_x=tl.sum((tl.abs(x)<float('inf')).to(tl.int32),axis=0)==32
    valid_w=tl.sum(((w>0)&(tl.abs(w)<float('inf'))).to(tl.int32),axis=0)==32
    valid_r=tl.sum(((r>0)&(tl.abs(r)<float('inf'))).to(tl.int32),axis=0)==32
    any_invalid=any_invalid | (~valid_x) | (~valid_w) | (~valid_r)
    if tl.sum(any_invalid.to(tl.int32),axis=0)>0:tl.atomic_or(ERROR,2)
    tl.store(CODES+offsets,best_code.to(tl.int8))
    tl.store(SIGNS+offsets,best_sign)
    tl.store(DECODED+offsets,best_decoded)
    metadata=group_id*128+column
    tl.store(SCALES+metadata,best_scale)
    tl.store(CHOICES+metadata,best_index.to(tl.uint8))
    tl.store(LOSSES+metadata,best_loss)


def _kernels():
    global _KERNELS
    if _KERNELS is None:
        from pathlib import Path
        from triton_runtime_guard import verify_before_import, verify_after_import
        root=Path(__file__).resolve().parents[1]
        verify_before_import(root)
        import triton
        verify_after_import(root,triton)
        import triton.language as language
        from triton.language.extra.cuda import libdevice as device_math
        globals()['tl']=language;globals()['libdevice']=device_math
        for fn in (_halves_add_first,_halves_sum_first):
            for name in ('REDUCTION','WIDTH'):fn.__annotations__[name]=language.constexpr
        globals()['_halves_add_first']=triton.jit(_halves_add_first)
        globals()['_halves_sum_first']=triton.jit(_halves_sum_first)
        for name in ('HEADS','OMEGA_BATCHED','A','B'):
            _row_kernel.__annotations__[name]=language.constexpr
        for name in ('HEADS','OMEGA_BATCHED','USE_ROW','SEARCH'):
            _group_kernel.__annotations__[name]=language.constexpr
        _KERNELS=(triton.jit(_row_kernel),triton.jit(_group_kernel))
    return _KERNELS


def runtime_dependency_receipt():
    from triton_runtime_guard import dependency_receipt
    return dependency_receipt()


def encode(x,condition,omega,a=None,b=None,*,debug=False,check_finite=True,**kwargs):
    import torch
    if kwargs:raise TypeError('unknown codec options: '+','.join(sorted(kwargs)))
    if condition not in METHODS:raise ValueError('unknown frozen condition')
    if not check_finite:raise ValueError('finite qualification may not be disabled')
    method=METHODS[condition]
    if x.ndim!=4 or tuple(x.shape[-2:])!=(128,128) or x.dtype!=torch.float32:
        raise ValueError('expected FP32 [B,H,128,128]')
    if x.device.type!='cuda' or not x.is_contiguous():
        raise ValueError('production codec requires contiguous CUDA FP32 carrier')
    batch,heads=x.shape[:2]
    if omega.dtype!=torch.float32 or omega.device!=x.device or not omega.is_contiguous():
        raise ValueError('omega must be contiguous FP32 on carrier CUDA device')
    if tuple(omega.shape) not in ((heads,128),(batch,heads,128)):
        raise ValueError('omega must bind [H,128] or [B,H,128] actual identities')
    use_row=method in ('rowbal','rowbal_wscale');search=method in ('wscale','rowbal_wscale')
    if use_row and (a,b) not in PARAMETER_GRID:
        raise ValueError('frozen CAL-selected row parameter pair required')
    row_kernel,group_kernel=_kernels()
    device=x.device
    codes=torch.empty((batch,heads,4,32,128),dtype=torch.int8,device=device)
    scales=torch.empty((batch,heads,4,128),dtype=torch.float32,device=device)
    signs=torch.empty_like(codes,dtype=torch.bool)
    factors=torch.empty((batch,heads,128),dtype=torch.float32,device=device) if use_row else \
        torch.ones((batch,heads,128),dtype=torch.float32,device=device)
    choices=torch.empty_like(scales,dtype=torch.uint8)
    losses=torch.empty_like(scales)
    decoded=torch.empty_like(x)
    invalid=torch.zeros((),dtype=torch.int32,device=device)
    common=dict(HEADS=heads,OMEGA_BATCHED=omega.ndim==3,num_warps=4,
                enable_fp_fusion=False)
    if use_row:row_kernel[(batch*heads*4,)](x,omega,factors,invalid,A=a,B=b,**common)
    group_kernel[(batch*heads*4*8,)](x,omega,factors,codes,scales,signs,choices,
        losses,decoded,invalid,USE_ROW=use_row,SEARCH=search,**common)
    e=Encoded(condition,tuple(x.shape),codes,scales,None,signs,factors,choices,losses,invalid,decoded)
    validate_encoded(e)  # once for entire layer-batched current token, before any writeback
    fields=dict(codes=codes,scales=scales,negative_zero_flags=signs,
        row_factors=factors,lambda_indices=choices,weighted_losses=losses,
        decoded=decoded,invalid_flags=invalid)
    return (e,fields) if debug else e
