def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.frame_id,self.frame_compile_id,self.compiled_autograd_id,)==(other.frame_id,other.frame_compile_id,other.compiled_autograd_id,)
  return NotImplemented
 return __eq__