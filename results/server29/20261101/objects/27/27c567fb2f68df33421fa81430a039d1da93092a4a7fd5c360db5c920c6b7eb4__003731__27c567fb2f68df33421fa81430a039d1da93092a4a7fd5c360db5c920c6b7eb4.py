def __create_fn__(_type_metadata, _type_tensors, _type_parameter_count, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _return_type):
 def __init__(self,metadata:_type_metadata,tensors:_type_tensors)->_return_type:
  self.metadata=metadata
  self.tensors=tensors
  self.__post_init__()
 return __init__