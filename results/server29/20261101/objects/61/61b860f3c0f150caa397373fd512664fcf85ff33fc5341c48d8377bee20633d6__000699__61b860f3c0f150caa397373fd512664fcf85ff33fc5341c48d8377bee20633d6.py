def __create_fn__(_type_actual, _type_ref, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _return_type):
 def __init__(self,actual:_type_actual,ref:_type_ref)->_return_type:
  __dataclass_builtins_object__.__setattr__(self,'actual',actual)
  __dataclass_builtins_object__.__setattr__(self,'ref',ref)
  self.__post_init__()
 return __init__