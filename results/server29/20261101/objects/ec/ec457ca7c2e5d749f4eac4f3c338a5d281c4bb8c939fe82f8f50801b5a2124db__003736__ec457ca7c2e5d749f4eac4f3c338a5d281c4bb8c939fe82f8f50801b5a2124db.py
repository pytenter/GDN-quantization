def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.metadata,self.sharded,self.weight_map,self.files_metadata,self.parameter_count,)==(other.metadata,other.sharded,other.weight_map,other.files_metadata,other.parameter_count,)
  return NotImplemented
 return __eq__