def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.match,self.shard_node,self.ag_node,self.res_node,self.gather_dim,self.group_name,)==(other.match,other.shard_node,other.ag_node,other.res_node,other.gather_dim,other.group_name,)
  return NotImplemented
 return __eq__