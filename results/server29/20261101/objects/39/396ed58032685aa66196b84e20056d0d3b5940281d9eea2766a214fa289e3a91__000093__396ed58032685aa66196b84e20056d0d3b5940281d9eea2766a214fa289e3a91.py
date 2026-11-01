def __create_fn__(_type_name, _type_arguments, _type_returns, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _return_type):
 def __init__(self,name:_type_name,arguments:_type_arguments,returns:_type_returns)->_return_type:
  __dataclass_builtins_object__.__setattr__(self,'name',name)
  __dataclass_builtins_object__.__setattr__(self,'arguments',arguments)
  __dataclass_builtins_object__.__setattr__(self,'returns',returns)
  self.__post_init__()
 return __init__