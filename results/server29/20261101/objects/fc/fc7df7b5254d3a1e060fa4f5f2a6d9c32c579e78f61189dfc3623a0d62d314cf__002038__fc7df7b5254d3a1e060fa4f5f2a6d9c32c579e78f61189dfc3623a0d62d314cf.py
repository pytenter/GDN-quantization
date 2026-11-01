def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.description,self.edge_traversals,self.immutable_state,self.initial_state,self.properties,self.result_graph_index,self.run_graph_index,)==(other.description,other.edge_traversals,other.immutable_state,other.initial_state,other.properties,other.result_graph_index,other.run_graph_index,)
  return NotImplemented
 return __eq__