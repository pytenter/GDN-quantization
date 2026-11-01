def __create_fn__(_type_reduce_op, _type_norm_type, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_reduce_op, _dflt_norm_type, _return_type):
 def __init__(self,reduce_op:_type_reduce_op=_dflt_reduce_op,norm_type:_type_norm_type=_dflt_norm_type)->_return_type:
  __dataclass_builtins_object__.__setattr__(self,'reduce_op',reduce_op)
  __dataclass_builtins_object__.__setattr__(self,'norm_type',norm_type)
  self.__post_init__()
 return __init__