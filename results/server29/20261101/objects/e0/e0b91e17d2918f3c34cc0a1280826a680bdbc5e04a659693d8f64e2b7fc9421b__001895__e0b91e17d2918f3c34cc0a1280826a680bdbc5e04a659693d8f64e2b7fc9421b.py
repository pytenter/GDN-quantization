def __create_fn__(_type_tensors, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_tensors, _return_type):
 def __init__(self,tensors:_type_tensors=_HAS_DEFAULT_FACTORY)->_return_type:
  self.tensors=_dflt_tensors() if tensors is _HAS_DEFAULT_FACTORY else tensors
 return __init__