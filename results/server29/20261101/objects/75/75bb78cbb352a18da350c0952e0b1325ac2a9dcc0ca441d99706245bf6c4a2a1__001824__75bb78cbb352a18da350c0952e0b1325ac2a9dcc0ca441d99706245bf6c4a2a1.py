def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.mesh,self.shard_mesh_dim,self.replicate_mesh_dim,)==(other.mesh,other.shard_mesh_dim,other.replicate_mesh_dim,)
  return NotImplemented
 return __eq__