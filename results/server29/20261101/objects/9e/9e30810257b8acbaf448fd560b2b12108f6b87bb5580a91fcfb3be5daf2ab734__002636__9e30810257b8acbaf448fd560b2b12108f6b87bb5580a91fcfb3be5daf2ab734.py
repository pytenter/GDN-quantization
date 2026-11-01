def __create_fn__(_type_dtype, _type_sizes, _type_requires_grad, _type_device, _type_strides, _type_storage_offset, _type_layout, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _return_type):
 def __init__(self,dtype:_type_dtype,sizes:_type_sizes,requires_grad:_type_requires_grad,device:_type_device,strides:_type_strides,storage_offset:_type_storage_offset,layout:_type_layout)->_return_type:
  self.dtype=dtype
  self.sizes=sizes
  self.requires_grad=requires_grad
  self.device=device
  self.strides=strides
  self.storage_offset=storage_offset
  self.layout=layout
 return __init__