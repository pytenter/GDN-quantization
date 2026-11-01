def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.dim,self.min,self.max,)==(other.dim,other.min,other.max,)
  return NotImplemented
 return __eq__