def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.match,self.input_node,self.rs_node,self.res_node,self.reduce_op,self.scatter_dim,self.group_name,)==(other.match,other.input_node,other.rs_node,other.res_node,other.reduce_op,other.scatter_dim,other.group_name,)
  return NotImplemented
 return __eq__