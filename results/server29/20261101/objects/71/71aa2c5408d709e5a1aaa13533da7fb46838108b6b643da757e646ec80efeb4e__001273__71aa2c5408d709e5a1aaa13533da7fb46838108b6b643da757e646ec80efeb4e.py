def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.allow_scalar_outputs,self.allow_dynamic_output_shape_ops,self.assume_static_by_default,self.specialize_zero_one,self.duck_shape,self.prefer_deferred_runtime_asserts_over_guards,self.allow_complex_guards_as_runtime_asserts,)==(other.allow_scalar_outputs,other.allow_dynamic_output_shape_ops,other.assume_static_by_default,other.specialize_zero_one,other.duck_shape,other.prefer_deferred_runtime_asserts_over_guards,other.allow_complex_guards_as_runtime_asserts,)
  return NotImplemented
 return __eq__