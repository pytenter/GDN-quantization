def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.parameters,self.total,)==(other.parameters,other.total,)
  return NotImplemented
 return __eq__