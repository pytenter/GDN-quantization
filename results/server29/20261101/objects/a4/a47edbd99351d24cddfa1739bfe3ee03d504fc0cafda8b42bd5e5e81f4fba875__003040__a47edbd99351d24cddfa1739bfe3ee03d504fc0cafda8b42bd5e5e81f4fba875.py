def __create_fn__(_type_name, _type_value, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _return_type):
 def __init__(self,*,name:_type_name,value:_type_value)->_return_type:
  __dataclass_builtins_object__.__setattr__(self,'name',name)
  __dataclass_builtins_object__.__setattr__(self,'value',value)
  self.__post_init__()
 return __init__