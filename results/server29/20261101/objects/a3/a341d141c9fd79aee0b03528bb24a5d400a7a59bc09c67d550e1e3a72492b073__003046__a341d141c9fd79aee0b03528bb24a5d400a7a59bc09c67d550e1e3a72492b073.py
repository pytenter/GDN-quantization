def __create_fn__(_type_name, _type_device, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _return_type):
 def __init__(self,*,name:_type_name,device:_type_device)->_return_type:
  __dataclass_builtins_object__.__setattr__(self,'name',name)
  __dataclass_builtins_object__.__setattr__(self,'device',device)
  self.__post_init__()
 return __init__