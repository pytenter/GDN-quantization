def __create_fn__(_type_device, _type_dtype, _type_inner_fn, _type_ranges, _type_output_indexer, _type_scatter_mode, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_scatter_mode, _return_type):
 def __init__(self,*,device:_type_device,dtype:_type_dtype,inner_fn:_type_inner_fn,ranges:_type_ranges,output_indexer:_type_output_indexer,scatter_mode:_type_scatter_mode=_dflt_scatter_mode)->_return_type:
  __dataclass_builtins_object__.__setattr__(self,'device',device)
  __dataclass_builtins_object__.__setattr__(self,'dtype',dtype)
  __dataclass_builtins_object__.__setattr__(self,'inner_fn',inner_fn)
  __dataclass_builtins_object__.__setattr__(self,'ranges',ranges)
  __dataclass_builtins_object__.__setattr__(self,'output_indexer',output_indexer)
  __dataclass_builtins_object__.__setattr__(self,'scatter_mode',scatter_mode)
  self.__post_init__()
 return __init__