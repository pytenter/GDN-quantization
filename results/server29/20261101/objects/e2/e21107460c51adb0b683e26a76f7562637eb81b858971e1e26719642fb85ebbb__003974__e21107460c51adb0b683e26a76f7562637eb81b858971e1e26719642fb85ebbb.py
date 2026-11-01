def __create_fn__(_type_backend, _type_init_method, _type_timeout, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_backend, _dflt_init_method, _dflt_timeout, _return_type):
 def __init__(self,backend:_type_backend=_dflt_backend,init_method:_type_init_method=_dflt_init_method,timeout:_type_timeout=_dflt_timeout)->_return_type:
  self.backend=backend
  self.init_method=init_method
  self.timeout=timeout
  self.__post_init__()
 return __init__