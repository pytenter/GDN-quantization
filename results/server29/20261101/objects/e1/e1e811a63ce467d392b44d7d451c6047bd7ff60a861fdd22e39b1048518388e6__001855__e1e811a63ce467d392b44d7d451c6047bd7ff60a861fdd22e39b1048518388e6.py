def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.tensors,self.scalar_tensors,self.non_tensors,)==(other.tensors,other.scalar_tensors,other.non_tensors,)
  return NotImplemented
 return __eq__