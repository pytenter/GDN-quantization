def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.tiling,self.score,self.name,)==(other.tiling,other.score,other.name,)
  return NotImplemented
 return __eq__