def __create_fn__(_type_flat_tensor_start_idx, _type_arg_count, _type_included_subclass_symints, _type_attrs, _type_outer_size, _type_outer_stride, _type_meta, _type_original_subclass, _type_original_subclass_type, _type_memory_format, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_original_subclass_type, _dflt_memory_format, _return_type):
 def __init__(self,flat_tensor_start_idx:_type_flat_tensor_start_idx,arg_count:_type_arg_count,included_subclass_symints:_type_included_subclass_symints,attrs:_type_attrs,outer_size:_type_outer_size,outer_stride:_type_outer_stride,meta:_type_meta,original_subclass:_type_original_subclass,original_subclass_type:_type_original_subclass_type=_dflt_original_subclass_type,memory_format:_type_memory_format=_dflt_memory_format)->_return_type:
  self.flat_tensor_start_idx=flat_tensor_start_idx
  self.arg_count=arg_count
  self.included_subclass_symints=included_subclass_symints
  self.attrs=attrs
  self.outer_size=outer_size
  self.outer_stride=outer_stride
  self.meta=meta
  self.original_subclass=original_subclass
  self.original_subclass_type=original_subclass_type
  self.memory_format=memory_format
  self.__post_init__()
 return __init__