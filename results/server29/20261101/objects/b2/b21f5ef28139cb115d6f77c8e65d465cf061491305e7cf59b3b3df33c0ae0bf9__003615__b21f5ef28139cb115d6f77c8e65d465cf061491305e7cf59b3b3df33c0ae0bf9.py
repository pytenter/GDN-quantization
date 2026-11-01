def __create_fn__(_type_msg, _type_kwargs, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _return_type):
 def __init__(self,*,msg:_type_msg,kwargs:_type_kwargs)->_return_type:
  __dataclass_builtins_object__.__setattr__(self,'msg',msg)
  __dataclass_builtins_object__.__setattr__(self,'kwargs',kwargs)
  self.__post_init__()
 return __init__