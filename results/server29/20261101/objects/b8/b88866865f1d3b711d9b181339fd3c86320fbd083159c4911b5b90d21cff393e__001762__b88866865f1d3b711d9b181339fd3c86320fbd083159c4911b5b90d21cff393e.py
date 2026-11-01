def __create_fn__(_type_reduce_op, _type_mask_buffer, _type_offset_shape, _type_offset_dim, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_reduce_op, _dflt_mask_buffer, _dflt_offset_shape, _dflt_offset_dim, _return_type):
 def __init__(self,reduce_op:_type_reduce_op=_dflt_reduce_op,mask_buffer:_type_mask_buffer=_HAS_DEFAULT_FACTORY,offset_shape:_type_offset_shape=_dflt_offset_shape,offset_dim:_type_offset_dim=_dflt_offset_dim)->_return_type:
  __dataclass_builtins_object__.__setattr__(self,'reduce_op',reduce_op)
  __dataclass_builtins_object__.__setattr__(self,'mask_buffer',_dflt_mask_buffer() if mask_buffer is _HAS_DEFAULT_FACTORY else mask_buffer)
  __dataclass_builtins_object__.__setattr__(self,'offset_shape',offset_shape)
  __dataclass_builtins_object__.__setattr__(self,'offset_dim',offset_dim)
 return __init__