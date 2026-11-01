def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.anchor,self.nodes_map,self.replacements,)==(other.anchor,other.nodes_map,other.replacements,)
  return NotImplemented
 return __eq__