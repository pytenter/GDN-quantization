def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.name,self.graph_module,self.graph,)==(other.name,other.graph_module,other.graph,)
  return NotImplemented
 return __eq__