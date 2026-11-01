"""Only an immutable INT8 byte string is carried between steps."""
import hashlib
import torch
from dual_axis_codec import NAMES,encode,EncodedState
from panel import thash,PARAMS

def run():
    result={}
    for codec in NAMES:
        shape=tuple(PARAMS['recurrent']['shape'])
        blob=encode(torch.zeros(shape),codec).to_bytes();snapshots=[]
        for step in range(1,129):
            stored=EncodedState.from_bytes(blob);old=stored.decode()
            g=torch.Generator(device='cpu').manual_seed(PARAMS['recurrent']['seed']+step)
            noise=torch.randn(shape,generator=g)
            k=torch.randn((*shape[:-1],1),generator=g);v=torch.randn((shape[0],shape[1],1,shape[-1]),generator=g)
            updated=0.97*old+0.01*noise+0.001*(k*v)
            new=encode(updated,codec);new_blob=new.to_bytes()
            # No float reconstruction, state or update enters the next step.
            del stored,old,noise,k,v,updated,new,g,blob
            blob=new_blob;del new_blob
            if step in (1,8,32,128):
                verify=EncodedState.from_bytes(blob)
                snapshots.append(dict(step=step,encoded_sha256=hashlib.sha256(blob).hexdigest(),decoded_sha256=thash(verify.decode()),bytes=len(blob),codes_dtype=str(verify.codes.dtype),next_step_state='encoded byte string only'))
                del verify
        result[codec]=snapshots
    return result
