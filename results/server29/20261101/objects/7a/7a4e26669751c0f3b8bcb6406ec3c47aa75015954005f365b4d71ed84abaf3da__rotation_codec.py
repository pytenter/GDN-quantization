"""P is exclusively a grouping permutation around the unchanged B1 codec."""
from dataclasses import dataclass
import torch
import group_codecs as base
P=tuple(int(f'{i:07b}'[::-1],2) for i in range(128))
@dataclass
class Encoded:
 base:object
 permuted:bool
 @property
 def codes(self):return self.base.codes
 @property
 def scales(self):return self.base.scales
 @property
 def zero_points(self):return self.base.zero_points
 @property
 def negative_zero_flags(self):return self.base.negative_zero_flags
 def cpu(self):return Encoded(self.base.cpu(),self.permuted)
def indices(device):return torch.tensor(P,dtype=torch.long,device=device)
def encode(x,method):
 if method not in ('P0','P1','C1','S3'):raise ValueError('Unknown frozen rotation condition')
 permuted=method in ('P1','S3')
 return Encoded(base.encode(x.index_select(-2,indices(x.device)) if permuted else x,'B1'),permuted)
def decode(e):
 d=base.decode(e.base)
 return d.index_select(-2,indices(d.device)) if e.permuted else d
def require_consumed(actual,e):
 expected=decode(e);cpu=actual.detach().cpu()
 if cpu.shape!=expected.shape or cpu.dtype!=expected.dtype or not torch.equal(cpu,expected) or base.tensor_bytes(cpu)!=base.tensor_bytes(expected):raise RuntimeError('Actual next helper initial_state differs from inverse-permuted QDQ writeback')
 return True
tensor_bytes=base.tensor_bytes
