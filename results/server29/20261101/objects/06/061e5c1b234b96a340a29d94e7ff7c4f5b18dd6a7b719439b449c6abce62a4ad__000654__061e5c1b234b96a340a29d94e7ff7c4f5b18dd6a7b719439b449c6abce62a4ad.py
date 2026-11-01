def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.dim,self.start,self.end,)==(other.dim,other.start,other.end,)
  return NotImplemented
 return __eq__