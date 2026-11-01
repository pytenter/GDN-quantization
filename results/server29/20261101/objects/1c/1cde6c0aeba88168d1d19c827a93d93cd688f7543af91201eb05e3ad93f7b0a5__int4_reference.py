"""Independent NumPy CPU scalar-operation reference; never imports production codec."""
import numpy as np

def reference(x):
    if x.dtype != np.float32 or x.ndim != 4 or x.shape[-2:] != (128,128):
        raise ValueError('reference layout/dtype')
    if not np.isfinite(x).all():
        raise FloatingPointError('nonfinite reference input')
    # Independent explicit Key slicing, rather than the production reshape.
    y = np.stack([x[:,:,k:k+32,:] for k in (0,32,64,96)],axis=2)
    z = np.zeros(y.shape[:-2] + (128,), dtype=np.float32)
    a = np.minimum(np.min(y, axis=-2), z)
    b = np.maximum(np.max(y, axis=-2), z)
    a = np.where(a == 0, z, a)
    b = np.where(b == 0, z, b)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        span = np.subtract(b,a,dtype=np.float32)
        safe = np.maximum(span, np.float32(1e-12))
        scale = np.divide(safe, np.full(safe.shape,15,dtype=np.float32),dtype=np.float32)
        if not np.isfinite(scale).all() or not (scale > 0).all():
            raise FloatingPointError('invalid reference FP32 scale')
        zero_ratio = np.divide(np.negative(a),scale,dtype=np.float32)
        zp = np.clip(np.rint(zero_ratio),0,15).astype(np.uint8)
        levels = np.divide(y,np.expand_dims(scale,-2),dtype=np.float32)
        if not np.isfinite(levels).all():
            raise FloatingPointError('nonfinite reference level')
        rounded = np.rint(levels)
        shifted = np.add(rounded,np.expand_dims(zp.astype(np.float32),-2),dtype=np.float32)
        codes = np.clip(shifted,0,15).astype(np.uint8)
        centered = np.subtract(codes.astype(np.int16),np.expand_dims(zp.astype(np.int16),-2),dtype=np.int16)
        decoded = np.multiply(centered.astype(np.float32),np.expand_dims(scale,-2),dtype=np.float32)
        decoded = np.where(decoded == 0,np.float32(0),decoded).reshape(x.shape)
        if not np.isfinite(decoded).all():
            raise FloatingPointError('nonfinite reference decode')
    return dict(minimum=a,maximum=b,span=span,safe_range=safe,scales=scale,
                zero_ratio=zero_ratio,zero_points=zp,levels=levels,rounded=rounded,
                shifted=shifted,codes=codes,decoded=decoded)

def fixtures():
    rng = np.random.default_rng(20261021)
    group = np.linspace(-7,8,32,dtype=np.float32)
    cases = {
        'ZERO': np.zeros(32,dtype=np.float32),
        'POS_CONSTANT': np.full(32,3,dtype=np.float32),
        'NEG_CONSTANT': np.full(32,-3,dtype=np.float32),
        'MIXED': group,
        'POS_OUTLIER': np.array([1e12]+[1]*31,dtype=np.float32),
        'NEG_OUTLIER': np.array([-1e12]+[-1]*31,dtype=np.float32),
        'SMALL_SPAN': np.linspace(-1e-18,1e-18,32,dtype=np.float32),
        'POS_SMALL_DYNAMIC': np.linspace(1,1.000001,32,dtype=np.float32),
        'TIES_ZP_ODD': np.array([-7,8,-6.5,-5.5,-.5,.5,1.5,2.5]*4,dtype=np.float32),
        'TIES_ZP0': np.array([0,15,.5,1.5,2.5,3.5,4.5,5.5]*4,dtype=np.float32),
        'NEG_ZP15': np.linspace(-15,0,32,dtype=np.float32),
        'FLOOR_EDGE': np.linspace(-5e-13,5e-13,32,dtype=np.float32),
        'EXTREME_FINITE': np.linspace(-np.finfo(np.float32).max/4,np.finfo(np.float32).max/4,32,dtype=np.float32),
        'SUBNORMAL_SIGNED_ZERO': np.array([0.,-0.,np.nextafter(np.float32(0),np.float32(1)),np.nextafter(np.float32(0),np.float32(-1))]*8,dtype=np.float32),
    }
    for name,g in cases.items():
        # Different value columns and different Key32 groups retain separateness.
        x=np.empty((1,2,128,128),dtype=np.float32)
        for h in range(2):
            for k in range(4):
                x[0,h,k*32:(k+1)*32,:]=g[:,None]
        yield name,x
    yield 'RANDOM_DISTINCT_GROUP_AXIS',rng.normal(size=(2,2,128,128)).astype(np.float32)

def rejection_fixtures():
    for name,values in [('NAN',[np.nan]),('POS_INF',[np.inf]),('NEG_INF',[-np.inf]),
                        ('FP32_SPAN_OVERFLOW',[-np.finfo(np.float32).max,np.finfo(np.float32).max])]:
        x=np.zeros((1,1,128,128),dtype=np.float32)
        x[0,0,:len(values),0]=values
        yield name,x
