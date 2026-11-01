def __create_fn__(_type_functional, _type_inplace, _type_mutable, _type_out, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _return_type):
 def __init__(self,functional:_type_functional,inplace:_type_inplace,mutable:_type_mutable,out:_type_out)->_return_type:
  __dataclass_builtins_object__.__setattr__(self,'functional',functional)
  __dataclass_builtins_object__.__setattr__(self,'inplace',inplace)
  __dataclass_builtins_object__.__setattr__(self,'mutable',mutable)
  __dataclass_builtins_object__.__setattr__(self,'out',out)
  self.__post_init__()
 return __init__