def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.dynamic_sizes,self.dynamic_strides,self.constraint_sizes,self.constraint_strides,self.view_base_context,self.tensor_source,self.shape_env_to_source_to_symbol_cache,self.inner_contexts,)==(other.dynamic_sizes,other.dynamic_strides,other.constraint_sizes,other.constraint_strides,other.view_base_context,other.tensor_source,other.shape_env_to_source_to_symbol_cache,other.inner_contexts,)
  return NotImplemented
 return __eq__