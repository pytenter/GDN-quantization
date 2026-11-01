def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.out_metas,self.fwd_output_strides,self.needs_post_compile,)==(other.out_metas,other.fwd_output_strides,other.needs_post_compile,)
  return NotImplemented
 return __eq__