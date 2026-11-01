def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.trace_joint,self.needs_post_compile,self.aliased_arg_idx_with_metadata_mutations,)==(other.trace_joint,other.needs_post_compile,other.aliased_arg_idx_with_metadata_mutations,)
  return NotImplemented
 return __eq__