def __create_fn__(_type_expr, _type_dtype, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _return_type):
 def __init__(self,expr:_type_expr,dtype:_type_dtype)->_return_type:
  self.expr=expr
  self.dtype=dtype
  self.__post_init__()
 return __init__