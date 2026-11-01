def __create_fn__(cls, FrozenInstanceError):
 def __delattr__(self,name):
  if type(self) is cls or name in ('allow_scalar_outputs','allow_dynamic_output_shape_ops','assume_static_by_default','specialize_zero_one','duck_shape','prefer_deferred_runtime_asserts_over_guards','allow_complex_guards_as_runtime_asserts',):
   raise FrozenInstanceError(f"cannot delete field {name!r}")
  super(cls, self).__delattr__(name)
 return __delattr__