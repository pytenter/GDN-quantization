def __create_fn__(_type_kind, _type_arg, _type_target, _type_persistent, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_persistent, _return_type):
 def __init__(self,kind:_type_kind,arg:_type_arg,target:_type_target,persistent:_type_persistent=_dflt_persistent)->_return_type:
  self.kind=kind
  self.arg=arg
  self.target=target
  self.persistent=persistent
  self.__post_init__()
 return __init__