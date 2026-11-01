def __create_fn__(_type_device, _type_size, _type_stride, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_size, _dflt_stride, _return_type):
 def __init__(self,*,device:_type_device,size:_type_size=_HAS_DEFAULT_FACTORY,stride:_type_stride=_HAS_DEFAULT_FACTORY)->_return_type:
  __dataclass_builtins_object__.__setattr__(self,'device',device)
  __dataclass_builtins_object__.__setattr__(self,'size',_dflt_size() if size is _HAS_DEFAULT_FACTORY else size)
  __dataclass_builtins_object__.__setattr__(self,'stride',_dflt_stride() if stride is _HAS_DEFAULT_FACTORY else stride)
 return __init__