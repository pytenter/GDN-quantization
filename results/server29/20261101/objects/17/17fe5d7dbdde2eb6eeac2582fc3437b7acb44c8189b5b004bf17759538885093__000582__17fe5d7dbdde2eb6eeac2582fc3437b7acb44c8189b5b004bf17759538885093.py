def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.anchors,self.nodes_map,self.placeholder_nodes,self.returning_nodes,self.name_node_map,)==(other.anchors,other.nodes_map,other.placeholder_nodes,other.returning_nodes,other.name_node_map,)
  return NotImplemented
 return __eq__