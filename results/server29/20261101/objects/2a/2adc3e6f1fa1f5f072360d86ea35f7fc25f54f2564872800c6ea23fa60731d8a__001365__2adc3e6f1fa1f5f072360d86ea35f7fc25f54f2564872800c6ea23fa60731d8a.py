def __create_fn__(_type_base, _type_member, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _return_type):
 def __init__(self,base:_type_base,member:_type_member)->_return_type:
  __dataclass_builtins_object__.__setattr__(self,'base',base)
  __dataclass_builtins_object__.__setattr__(self,'member',member)
  self.__post_init__()
 return __init__