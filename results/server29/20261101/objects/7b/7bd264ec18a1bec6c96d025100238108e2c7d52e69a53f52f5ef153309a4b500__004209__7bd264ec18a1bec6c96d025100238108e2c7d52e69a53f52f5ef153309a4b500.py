def __create_fn__(_type_is_sharded, _type_metadata, _type_filename_to_tensors, _type_tensor_to_filename, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _return_type):
 def __init__(self,metadata:_type_metadata,filename_to_tensors:_type_filename_to_tensors,tensor_to_filename:_type_tensor_to_filename)->_return_type:
  self.metadata=metadata
  self.filename_to_tensors=filename_to_tensors
  self.tensor_to_filename=tensor_to_filename
  self.__post_init__()
 return __init__