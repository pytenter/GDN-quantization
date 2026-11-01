def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.nodes,self.arg_ancestor_nodes,self.A_node,self.B_node,)==(other.nodes,other.arg_ancestor_nodes,other.A_node,other.B_node,)
  return NotImplemented
 return __eq__