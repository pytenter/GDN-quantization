def __create_fn__(_type_dtype, _type_layout, _type_requires_grad, _type_memory_format, _type_pin_memory, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_dtype, _dflt_layout, _dflt_requires_grad, _dflt_memory_format, _dflt_pin_memory, _return_type):
 def __init__(self,dtype:_type_dtype=_dflt_dtype,layout:_type_layout=_dflt_layout,requires_grad:_type_requires_grad=_dflt_requires_grad,memory_format:_type_memory_format=_dflt_memory_format,pin_memory:_type_pin_memory=_dflt_pin_memory)->_return_type:
  self.dtype=dtype
  self.layout=layout
  self.requires_grad=requires_grad
  self.memory_format=memory_format
  self.pin_memory=pin_memory
 return __init__