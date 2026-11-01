def __create_fn__(_type_scalar, _type_size, _type_stride, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_scalar, _dflt_size, _dflt_stride, _return_type):
 def __init__(self,scalar:_type_scalar=_dflt_scalar,size:_type_size=_dflt_size,stride:_type_stride=_dflt_stride)->_return_type:
  self.scalar=scalar
  self.size=size
  self.stride=stride
  self.__post_init__()
 return __init__