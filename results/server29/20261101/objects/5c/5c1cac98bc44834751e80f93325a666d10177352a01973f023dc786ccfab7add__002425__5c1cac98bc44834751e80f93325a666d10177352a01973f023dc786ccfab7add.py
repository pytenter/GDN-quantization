def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.tensor_metadata,self.values,)==(other.tensor_metadata,other.values,)
  return NotImplemented
 return __eq__