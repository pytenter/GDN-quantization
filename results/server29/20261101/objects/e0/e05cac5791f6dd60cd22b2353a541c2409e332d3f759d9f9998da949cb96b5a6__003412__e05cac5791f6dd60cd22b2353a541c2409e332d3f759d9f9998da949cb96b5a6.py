def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.nodes,self.arg_ancestor_nodes,self.A_node,self.B_node,self.A_scale_node,self.B_scale_node,self.bias_node,self.result_scale_node,self.out_dtype,self.use_fast_accum,)==(other.nodes,other.arg_ancestor_nodes,other.A_node,other.B_node,other.A_scale_node,other.B_scale_node,other.bias_node,other.result_scale_node,other.out_dtype,other.use_fast_accum,)
  return NotImplemented
 return __eq__