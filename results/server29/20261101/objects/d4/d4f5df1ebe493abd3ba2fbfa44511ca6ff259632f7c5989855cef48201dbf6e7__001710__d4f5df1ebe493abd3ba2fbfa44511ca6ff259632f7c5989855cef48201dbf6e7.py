def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.local_rank,self.global_rank,self.local_world_size,self.global_world_size,)==(other.local_rank,other.global_rank,other.local_world_size,other.global_world_size,)
  return NotImplemented
 return __eq__