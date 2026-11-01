def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.graph,self.block,self.opset,self.original_node,self.params_dict,self.env,self.values_in_env,self.new_nodes,)==(other.graph,other.block,other.opset,other.original_node,other.params_dict,other.env,other.values_in_env,other.new_nodes,)
  return NotImplemented
 return __eq__