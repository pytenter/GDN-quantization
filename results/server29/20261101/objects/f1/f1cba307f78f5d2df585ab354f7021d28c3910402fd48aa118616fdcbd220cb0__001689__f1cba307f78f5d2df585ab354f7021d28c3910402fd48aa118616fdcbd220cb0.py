def __create_fn__(_type_shards_metadata, _type_size, _type_tensor_properties, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_shards_metadata, _dflt_size, _dflt_tensor_properties, _return_type):
 def __init__(self,shards_metadata:_type_shards_metadata=_HAS_DEFAULT_FACTORY,size:_type_size=_dflt_size,tensor_properties:_type_tensor_properties=_HAS_DEFAULT_FACTORY)->_return_type:
  self.shards_metadata=_dflt_shards_metadata() if shards_metadata is _HAS_DEFAULT_FACTORY else shards_metadata
  self.size=size
  self.tensor_properties=_dflt_tensor_properties() if tensor_properties is _HAS_DEFAULT_FACTORY else tensor_properties
 return __init__