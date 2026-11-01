def __create_fn__(_type_device, _type_dtype, _type_inner_fn, _type_ranges, _type_reduction_ranges, _type_reduction_type, _type_src_dtype, _type_reduction_hint, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _return_type):
 def __init__(self,*,device:_type_device,dtype:_type_dtype,inner_fn:_type_inner_fn,ranges:_type_ranges,reduction_ranges:_type_reduction_ranges,reduction_type:_type_reduction_type,src_dtype:_type_src_dtype,reduction_hint:_type_reduction_hint)->_return_type:
  __dataclass_builtins_object__.__setattr__(self,'device',device)
  __dataclass_builtins_object__.__setattr__(self,'dtype',dtype)
  __dataclass_builtins_object__.__setattr__(self,'inner_fn',inner_fn)
  __dataclass_builtins_object__.__setattr__(self,'ranges',ranges)
  __dataclass_builtins_object__.__setattr__(self,'reduction_ranges',reduction_ranges)
  __dataclass_builtins_object__.__setattr__(self,'reduction_type',reduction_type)
  __dataclass_builtins_object__.__setattr__(self,'src_dtype',src_dtype)
  __dataclass_builtins_object__.__setattr__(self,'reduction_hint',reduction_hint)
  self.__post_init__()
 return __init__