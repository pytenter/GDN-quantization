def __create_fn__(cls, FrozenInstanceError):
 def __delattr__(self,name):
  if type(self) is cls or name in ('dynamic_sizes','dynamic_strides','constraint_sizes','constraint_strides','view_base_context','tensor_source','shape_env_to_source_to_symbol_cache','inner_contexts',):
   raise FrozenInstanceError(f"cannot delete field {name!r}")
  super(cls, self).__delattr__(name)
 return __delattr__