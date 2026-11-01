def __create_fn__(_type_base, _type_idx_key, _type_is_kw, _type_field, _type__name, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_is_kw, _return_type):
 def __init__(self,base:_type_base,idx_key:_type_idx_key,is_kw:_type_is_kw=_dflt_is_kw)->_return_type:
  __dataclass_builtins_object__.__setattr__(self,'base',base)
  __dataclass_builtins_object__.__setattr__(self,'idx_key',idx_key)
  __dataclass_builtins_object__.__setattr__(self,'is_kw',is_kw)
  self.__post_init__()
 return __init__