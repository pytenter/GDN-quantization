def __create_fn__(_type_base, _type_prop, _type_idx, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_idx, _return_type):
 def __init__(self,base:_type_base,prop:_type_prop,idx:_type_idx=_dflt_idx)->_return_type:
  __dataclass_builtins_object__.__setattr__(self,'base',base)
  __dataclass_builtins_object__.__setattr__(self,'prop',prop)
  __dataclass_builtins_object__.__setattr__(self,'idx',idx)
  self.__post_init__()
 return __init__