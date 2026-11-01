def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.shard_offsets,self.shard_sizes,self.placement,)==(other.shard_offsets,other.shard_sizes,other.placement,)
  return NotImplemented
 return __eq__