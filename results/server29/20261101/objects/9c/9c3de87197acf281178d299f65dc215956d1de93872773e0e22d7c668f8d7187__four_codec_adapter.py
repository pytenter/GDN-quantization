"""Compatibility name for the unchanged qualified Engine import contract."""
from unified_adapter import UnifiedQDQ


class NativeLayerTracker:
    """Original coordinate-free layer identification hooks, unchanged semantics."""
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


FourCodecQDQ = UnifiedQDQ
