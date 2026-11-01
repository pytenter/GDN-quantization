def __create_fn__(_type_replace, _type_add, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _return_type):
 def __init__(self,replace:_type_replace,add:_type_add)->_return_type:
  __dataclass_builtins_object__.__setattr__(self,'replace',replace)
  __dataclass_builtins_object__.__setattr__(self,'add',add)
  self.__post_init__()
 return __init__