def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.type,self.context,self.children_specs,self.num_nodes,self.num_leaves,self.num_children,)==(other.type,other.context,other.children_specs,other.num_nodes,other.num_leaves,other.num_children,)
  return NotImplemented
 return __eq__