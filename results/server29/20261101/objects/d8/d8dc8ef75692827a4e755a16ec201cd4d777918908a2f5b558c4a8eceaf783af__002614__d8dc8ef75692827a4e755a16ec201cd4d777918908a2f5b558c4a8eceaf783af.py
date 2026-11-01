def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.graph,self.inps,)==(other.graph,other.inps,)
  return NotImplemented
 return __eq__