def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.name,self.expr,)==(other.name,other.expr,)
  return NotImplemented
 return __eq__