def __create_fn__(_type_data, _type_target_dtype, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _return_type):
 def __init__(self,*,data:_type_data,target_dtype:_type_target_dtype)->_return_type:
  __dataclass_builtins_object__.__setattr__(self,'data',data)
  __dataclass_builtins_object__.__setattr__(self,'target_dtype',target_dtype)
  self.__post_init__()
 return __init__