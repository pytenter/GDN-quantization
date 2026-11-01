def __create_fn__(_type_dtype, _type_shape, _type_data_offsets, _type_parameter_count, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _return_type):
 def __init__(self,dtype:_type_dtype,shape:_type_shape,data_offsets:_type_data_offsets)->_return_type:
  self.dtype=dtype
  self.shape=shape
  self.data_offsets=data_offsets
  self.__post_init__()
 return __init__