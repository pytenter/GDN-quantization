def __create_fn__(_type_metadata, _type_sharded, _type_weight_map, _type_files_metadata, _type_parameter_count, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _return_type):
 def __init__(self,metadata:_type_metadata,sharded:_type_sharded,weight_map:_type_weight_map,files_metadata:_type_files_metadata)->_return_type:
  self.metadata=metadata
  self.sharded=sharded
  self.weight_map=weight_map
  self.files_metadata=files_metadata
  self.__post_init__()
 return __init__