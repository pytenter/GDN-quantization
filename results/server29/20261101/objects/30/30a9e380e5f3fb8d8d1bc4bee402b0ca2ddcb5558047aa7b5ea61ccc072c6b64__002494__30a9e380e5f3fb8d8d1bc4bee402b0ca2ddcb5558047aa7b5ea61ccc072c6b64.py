def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.keep_arg_mask,self.add_dupe_map,self.old_input_metadata,self.needs_post_compile,)==(other.keep_arg_mask,other.add_dupe_map,other.old_input_metadata,other.needs_post_compile,)
  return NotImplemented
 return __eq__