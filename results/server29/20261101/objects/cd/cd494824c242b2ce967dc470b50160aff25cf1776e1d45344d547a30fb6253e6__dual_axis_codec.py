"""CPU-only physical signed INT8 codecs; no float state survives encode()."""
from dataclasses import dataclass
import struct
import torch
from references import CANON

NAMES = ('C128_CANONICAL','C128_FP16_SCALE','C128_LS_FP16','DUAL_M_INT8')
HEADER = struct.Struct('<4sBBBB4I8x')  # exactly 32 on-wire bytes, version 1, little endian
LOW, HIGH = 2.0**-24, 65504.0

def safe_half(x):
    if torch.isnan(x).any(): raise ValueError('NaN fitted scale')
    return x.clamp(LOW,HIGH).to(torch.float16).contiguous().clone()

def source(x):
    if not isinstance(x,torch.Tensor) or x.device.type!='cpu' or x.ndim!=4 or min(x.shape)<1 or not x.is_floating_point():
        raise ValueError('Require a nonempty floating CPU [B,H,K,V] state')
    if not torch.isfinite(x).all(): raise ValueError('Nonfinite input')
    x=x.detach().float()
    if not torch.isfinite(x).all(): raise ValueError('Nonfinite FP32 input conversion')
    return x

def levels(y):
    if torch.isnan(y).any():raise ValueError('NaN levels')
    return torch.round(y).clamp(-127,127)

@dataclass(frozen=True)
class EncodedState:
    codec: str
    shape: tuple
    codes: torch.Tensor
    columns: torch.Tensor
    rows: torch.Tensor | None = None

    def validate(self):
        if self.codec not in NAMES or len(self.shape)!=4 or min(self.shape)<1:raise ValueError('Bad metadata')
        b,h,k,v=self.shape
        if self.codes.shape!=self.shape or self.codes.dtype!=torch.int8 or (self.codes==-128).any(): raise ValueError('Invalid signed INT8 codes')
        dtype=torch.float32 if self.codec==NAMES[0] else torch.float16
        if self.columns.shape!=(b,h,1,v) or self.columns.dtype!=dtype: raise ValueError('Bad columns')
        tensors=[self.codes,self.columns]
        if self.codec==NAMES[3]:
            if self.rows is None or self.rows.shape!=(b,h,k,1) or self.rows.dtype!=torch.float16: raise ValueError('Bad rows')
            tensors.append(self.rows)
        elif self.rows is not None: raise ValueError('Unexpected rows')
        for t in tensors:
            if t.device.type!='cpu' or not t.is_contiguous() or t._base is not None or t.untyped_storage().nbytes()!=t.numel()*t.element_size(): raise ValueError('Storage must be CPU, owned and exact sized')
        for t in tensors[1:]:
            if not torch.isfinite(t).all() or not (t>0).all():raise ValueError('Bad scale values')
        return self

    def decode(self):
        # Same operation order as official fit_state; StateCodec.decode uses a different association.
        if self.rows is not None:return (self.rows.float()*self.columns.float())*self.codes.float()
        return self.codes.float()*self.columns.float()

    def byte_audit(self):
        code=self.codes.untyped_storage().nbytes(); col=self.columns.untyped_storage().nbytes()
        row=0 if self.rows is None else self.rows.untyped_storage().nbytes()
        return dict(code_bytes=code,row_scale_bytes=row,column_scale_bytes=col,metadata_bytes=HEADER.size,
                    tensor_payload_bytes=code+row+col,total_physical_payload_bytes=HEADER.size+code+row+col,
                    effective_bits_per_value=8*(HEADER.size+code+row+col)/self.codes.numel(),
                    tensor_only_bits_per_value=8*(code+row+col)/self.codes.numel(),
                    host_python_allocator_object_overhead='NOT_INCLUDED; wire payload and actual tensor storages measured')

    def to_bytes(self):
        self.validate()
        header=HEADER.pack(b'QDAX',1,NAMES.index(self.codec),4,0,*self.shape)
        payload=self.codes.numpy().tobytes()+self.columns.numpy().tobytes()
        if self.rows is not None:payload+=self.rows.numpy().tobytes()
        return header+payload

    @classmethod
    def from_bytes(cls, blob):
        if len(blob)<HEADER.size:raise ValueError('Truncated header')
        magic,version,idx,rank,reserved,b,h,k,v=HEADER.unpack(blob[:HEADER.size])
        if magic!=b'QDAX' or version!=1 or idx>=len(NAMES) or rank!=4 or reserved or min(b,h,k,v)<1:raise ValueError('Bad header')
        count=b*h*k*v; nc=b*h*v; nr=b*h*k if idx==3 else 0
        dtype=torch.float32 if idx==0 else torch.float16; width=4 if idx==0 else 2
        expected=HEADER.size+count+nc*width+nr*2
        if len(blob)!=expected:raise ValueError('Wrong payload length')
        off=HEADER.size
        def take(n,d):
            nonlocal off
            size=n*torch.empty((),dtype=d).element_size()
            t=torch.frombuffer(bytearray(blob[off:off+size]),dtype=d).clone();off+=size
            return t
        z=take(count,torch.int8).reshape(b,h,k,v).clone()
        c=take(nc,dtype).reshape(b,h,1,v).clone()
        r=take(nr,torch.float16).reshape(b,h,k,1).clone() if nr else None
        return cls(NAMES[idx],(b,h,k,v),z,c,r).validate()

def encode(x, codec):
    x=source(x)
    if codec not in NAMES:raise ValueError('Unknown codec')
    if codec==NAMES[0]:
        # Calls actual frozen reference, not a replacement formula.
        ref=CANON['qwen_c128_qdq'](x)
        # Match the guard in the actual formal writeback path; never persist Inf.
        if not torch.isfinite(ref.dequant).all():raise FloatingPointError('nonfinite C128 dequant')
        return EncodedState(codec,tuple(x.shape),ref.codes.to(torch.int8).contiguous().clone(),ref.scale.contiguous().clone()).validate()
    if codec in NAMES[1:3]:
        c=safe_half(x.abs().amax(-2,keepdim=True).clamp_min(CANON['EPS'])/127.0)
        z=levels(x/c.float())
        if codec==NAMES[2]:
            den=z.square().sum(-2,keepdim=True)
            c=safe_half(torch.where(den>0,(z*x).sum(-2,keepdim=True)/den.clamp_min(1e-30),c.float()))
            z=levels(x/c.float())
        return EncodedState(codec,tuple(x.shape),z.to(torch.int8).contiguous().clone(),c).validate()
    r=safe_half(x.abs().mean(-1,keepdim=True).clamp_min(1e-20).sqrt())
    scaled=x/r.float()
    c=safe_half((scaled.abs()/127.0).amax(-2,keepdim=True))
    z=levels(scaled/c.float())
    v=r.float()*z
    den=v.square().sum(-2,keepdim=True)
    c=safe_half(torch.where(den>0,(v*x).sum(-2,keepdim=True)/den.clamp_min(1e-30),c.float()))
    z=levels(scaled/c.float())
    return EncodedState(codec,tuple(x.shape),z.to(torch.int8).contiguous().clone(),c,r).validate()
