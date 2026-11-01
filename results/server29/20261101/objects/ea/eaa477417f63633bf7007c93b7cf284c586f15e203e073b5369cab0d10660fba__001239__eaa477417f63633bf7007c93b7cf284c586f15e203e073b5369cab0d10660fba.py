def __create_fn__(cls, FrozenInstanceError):
 def __setattr__(self,name,value):
  if type(self) is cls or name in ('dynamic_sizes','dynamic_strides','constraint_sizes','constraint_strides','view_base_context','tensor_source','shape_env_to_source_to_symbol_cache',):
   raise FrozenInstanceError(f"cannot assign to field {name!r}")
  super(cls, self).__setattr__(name, value)
 return __setattr__