def __create_fn__(_type_flat_requires_grad, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_flat_requires_grad, _return_type):
 def __init__(self,flat_requires_grad:_type_flat_requires_grad=_HAS_DEFAULT_FACTORY)->_return_type:
  self.flat_requires_grad=_dflt_flat_requires_grad() if flat_requires_grad is _HAS_DEFAULT_FACTORY else flat_requires_grad
 return __init__