def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.sym_node_lookup,self.shape_env,)==(other.sym_node_lookup,other.shape_env,)
  return NotImplemented
 return __eq__