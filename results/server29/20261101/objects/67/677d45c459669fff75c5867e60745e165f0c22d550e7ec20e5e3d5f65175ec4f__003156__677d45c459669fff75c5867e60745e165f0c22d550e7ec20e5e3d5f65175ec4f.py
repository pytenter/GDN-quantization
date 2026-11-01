def __create_fn__(_type_device, _type_dtype, _type_sizes, _type_strides, _type_offset, _type_name, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_name, _return_type):
 def __init__(self,device:_type_device,dtype:_type_dtype,sizes:_type_sizes,strides:_type_strides,offset:_type_offset,name:_type_name=_dflt_name)->_return_type:
  self.device=device
  self.dtype=dtype
  self.sizes=sizes
  self.strides=strides
  self.offset=offset
  self.name=name
 return __init__