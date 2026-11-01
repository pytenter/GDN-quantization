def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.is_sharded,self.metadata,self.filename_to_tensors,self.tensor_to_filename,)==(other.is_sharded,other.metadata,other.filename_to_tensors,other.tensor_to_filename,)
  return NotImplemented
 return __eq__