def __create_fn__(_type_tensor, _type_metadata, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _return_type):
 def __init__(self,tensor:_type_tensor,metadata:_type_metadata)->_return_type:
  self.tensor=tensor
  self.metadata=metadata
  self.__post_init__()
 return __init__