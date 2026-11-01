def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.top_node_idx,self.nodes,self.inputs,self.nodes_need_process,)==(other.top_node_idx,other.nodes,other.inputs,other.nodes_need_process,)
  return NotImplemented
 return __eq__