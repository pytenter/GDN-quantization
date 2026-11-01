"""Independent NumPy reference: explicit Key-group index slicing, no torch/codecs."""
import numpy as np

F = np.float32

def reference(x, method, semantics='CPU_CANONICAL'):
    x = np.asarray(x,dtype=F)
    b,h,k,v=x.shape
    assert k==v==128 and np.isfinite(x).all()
    size=128 if method=='B0' else 32
    out=np.empty_like(x);scales=np.empty((b,h,k//size,v),F)
    codes=np.empty((b,h,k//size,size,v),np.uint8 if method=='B2' else np.int8)
    zps=np.empty_like(scales,dtype=np.uint8) if method=='B2' else None
    nz=np.empty_like(codes,dtype=bool) if method!='B2' else None
    for bi in range(b):
        for hi in range(h):
            for gi,start in enumerate(range(0,k,size)):
                g=x[bi,hi,start:start+size,:]
                if method!='B2':
                    peak=np.max(np.abs(g),axis=0)
                    safe=np.maximum(peak,F(1e-12))
                    s=F(safe*F(1/127)) if semantics=='CUDA_LEGACY_EMULATED' else F(safe/F(127))
                    r=np.rint(F(g/s))
                    c=np.clip(r,-127,127).astype(np.int8)
                    negativezero=(r==0)&np.signbit(r)
                    d=F(c.astype(F)*s)
                    d[negativezero]=F(-0.)
                    nz[bi,hi,gi]=negativezero
                else:
                    a=np.minimum(np.min(g,axis=0),F(0));a[a==0]=F(0)
                    top=np.maximum(np.max(g,axis=0),F(0));top[top==0]=F(0)
                    span=F(top-a)
                    s=F(np.maximum(span,F(1e-12))/F(255))
                    z=np.clip(np.rint(F(-a/s)),0,255).astype(np.uint8)
                    r=np.rint(F(g/s))
                    c=np.clip(F(r+z.astype(F)),0,255).astype(np.uint8)
                    d=F((c.astype(np.int16)-z.astype(np.int16)).astype(F)*s)
                    d[d==0]=F(0)
                    zps[bi,hi,gi]=z
                scales[bi,hi,gi]=s;codes[bi,hi,gi]=c
                out[bi,hi,start:start+size,:]=d
    return dict(codes=codes,scales=scales,zero_points=zps,negative_zero_flags=nz,decoded=out)
