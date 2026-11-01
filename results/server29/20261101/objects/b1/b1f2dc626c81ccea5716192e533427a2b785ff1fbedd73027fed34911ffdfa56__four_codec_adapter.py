"""Coordinate-only addition around the byte-identical qualified Group32 codec."""
from pathlib import Path
import sys
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'inherited/group32/src'))
from persistent_adapter import PersistentQDQ
from canonical_h128 import get_state
CONDITIONS={'G64-0':('NATIVE','B1'),'G64-1':('NATIVE','B2'),'G64-2':('H128','B1'),'G64-3':('H128','B2')}

class NativeLayerTracker:
    """Hooks identify the layer only. No q/k transform or helper replacement."""
    def __init__(self,model):
        self.model=model;self.handles=[];self.current_layer=None
    def __enter__(self):
        layers=self.model.model.layers if hasattr(self.model.model,'layers') else self.model.model.language_model.layers
        for i,layer in enumerate(layers):
            module=getattr(layer,'linear_attn',None) or getattr(layer,'self_attn',None)
            if module is not None:self.handles.append(module.register_forward_pre_hook(lambda _m,_a,j=i:setattr(self,'current_layer',j)))
        return self
    def __exit__(self,*args):
        for h in self.handles:h.remove()

class FourCodecQDQ(PersistentQDQ):
    def __init__(self,model,condition,layers,observe=True,callback=None):
        if condition not in CONDITIONS:raise ValueError('Unknown frozen codec')
        coordinate,legacy=CONDITIONS[condition]
        super().__init__(model,legacy,layers,observe,callback,group_size=64)
        self.condition=condition;self.coordinate=coordinate
        if coordinate=='NATIVE':self.h=NativeLayerTracker(model)
    def audit(self):
        return dict(super().audit(),condition=self.condition,coordinate=self.coordinate,
                    codec_shared_parameterized=True,group_axis='Key',group_size=64)
