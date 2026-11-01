def __create_fn__():
 def __hash__(self):
  return hash((self.shape,self.node_list,self.inputs,self.wait_nodes,self.comm_node,self.outputs,))
 return __hash__