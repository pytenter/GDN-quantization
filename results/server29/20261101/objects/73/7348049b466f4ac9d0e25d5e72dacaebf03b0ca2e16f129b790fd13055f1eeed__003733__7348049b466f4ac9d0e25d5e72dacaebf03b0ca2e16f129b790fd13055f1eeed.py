def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.metadata,self.tensors,self.parameter_count,)==(other.metadata,other.tensors,other.parameter_count,)
  return NotImplemented
 return __eq__