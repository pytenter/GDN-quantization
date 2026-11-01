def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.inst,self.target,self.stack_index,self.with_context,)==(other.inst,other.target,other.stack_index,other.with_context,)
  return NotImplemented
 return __eq__