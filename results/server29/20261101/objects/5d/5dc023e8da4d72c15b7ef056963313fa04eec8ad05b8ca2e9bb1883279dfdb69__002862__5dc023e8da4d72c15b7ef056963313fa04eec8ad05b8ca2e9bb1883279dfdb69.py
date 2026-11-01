def __create_fn__(_type_device, _type_dtype, _type_inner_fn, _type_ranges, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _return_type):
 def __init__(self,*,device:_type_device,dtype:_type_dtype,inner_fn:_type_inner_fn,ranges:_type_ranges)->_return_type:
  __dataclass_builtins_object__.__setattr__(self,'device',device)
  __dataclass_builtins_object__.__setattr__(self,'dtype',dtype)
  __dataclass_builtins_object__.__setattr__(self,'inner_fn',inner_fn)
  __dataclass_builtins_object__.__setattr__(self,'ranges',ranges)
  self.__post_init__()
 return __init__