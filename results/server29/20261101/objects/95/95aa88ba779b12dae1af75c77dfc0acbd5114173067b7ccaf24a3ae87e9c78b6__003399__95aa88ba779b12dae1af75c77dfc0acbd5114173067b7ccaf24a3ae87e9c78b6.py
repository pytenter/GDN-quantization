def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.shape,self.node_list,self.inputs,self.wait_nodes,self.comm_node,self.outputs,)==(other.shape,other.node_list,other.inputs,other.wait_nodes,other.comm_node,other.outputs,)
  return NotImplemented
 return __eq__