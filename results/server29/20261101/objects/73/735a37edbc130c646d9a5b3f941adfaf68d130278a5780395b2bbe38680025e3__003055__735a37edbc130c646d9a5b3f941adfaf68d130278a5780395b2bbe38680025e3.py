def __create_fn__(_type_value, _type_is_symbolic, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_is_symbolic, _return_type):
 def __init__(self,value:_type_value,is_symbolic:_type_is_symbolic=_dflt_is_symbolic)->_return_type:
  self.value=value
  self.is_symbolic=is_symbolic
  self.__post_init__()
 return __init__