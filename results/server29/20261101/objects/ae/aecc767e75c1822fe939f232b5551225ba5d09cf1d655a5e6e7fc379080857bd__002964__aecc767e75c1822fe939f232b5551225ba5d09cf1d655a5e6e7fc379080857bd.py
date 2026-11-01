def __create_fn__(_type_dtype, _type_device, _type_index, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _return_type):
 def __init__(self,*,dtype:_type_dtype,device:_type_device,index:_type_index)->_return_type:
  __dataclass_builtins_object__.__setattr__(self,'dtype',dtype)
  __dataclass_builtins_object__.__setattr__(self,'device',device)
  __dataclass_builtins_object__.__setattr__(self,'index',index)
  self.__post_init__()
 return __init__