def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.shards_metadata,self.size,self.tensor_properties,)==(other.shards_metadata,other.size,other.tensor_properties,)
  return NotImplemented
 return __eq__