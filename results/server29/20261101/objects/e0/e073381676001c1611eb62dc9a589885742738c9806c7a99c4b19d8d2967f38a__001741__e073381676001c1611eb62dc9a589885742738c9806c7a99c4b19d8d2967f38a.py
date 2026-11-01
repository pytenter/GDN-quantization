def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.mesh,self.placements,self.tensor_meta,)==(other.mesh,other.placements,other.tensor_meta,)
  return NotImplemented
 return __eq__