def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.graph,self.name,self.description,)==(other.graph,other.name,other.description,)
  return NotImplemented
 return __eq__