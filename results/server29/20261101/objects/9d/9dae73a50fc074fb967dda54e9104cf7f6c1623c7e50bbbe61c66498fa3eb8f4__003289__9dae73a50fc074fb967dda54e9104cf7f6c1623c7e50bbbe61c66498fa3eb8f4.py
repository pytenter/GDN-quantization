def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.node,self.symbol,self.attr,self.dim,)==(other.node,other.symbol,other.attr,other.dim,)
  return NotImplemented
 return __eq__