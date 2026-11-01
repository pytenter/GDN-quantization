def __create_fn__(_type_size, _type_sha256, _type_pointer_size, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _return_type):
 def __init__(self,size:_type_size,sha256:_type_sha256,pointer_size:_type_pointer_size)->_return_type:
  self.size=size
  self.sha256=sha256
  self.pointer_size=pointer_size
  self.__post_init__()
 return __init__