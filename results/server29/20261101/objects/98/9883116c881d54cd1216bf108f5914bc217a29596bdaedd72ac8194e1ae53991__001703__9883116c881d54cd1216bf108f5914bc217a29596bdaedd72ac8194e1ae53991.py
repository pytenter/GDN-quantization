def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.dim,self.placements,)==(other.dim,other.placements,)
  return NotImplemented
 return __eq__