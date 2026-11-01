def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.rank,self.world_size,self.local_rank,self.local_world_size,)==(other.rank,other.world_size,other.local_rank,other.local_world_size,)
  return NotImplemented
 return __eq__