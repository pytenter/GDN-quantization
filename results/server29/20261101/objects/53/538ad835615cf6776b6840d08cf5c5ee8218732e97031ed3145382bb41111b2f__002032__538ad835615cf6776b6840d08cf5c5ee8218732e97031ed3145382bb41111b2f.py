def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.description,self.edges,self.nodes,self.properties,)==(other.description,other.edges,other.nodes,other.properties,)
  return NotImplemented
 return __eq__