def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.dtype,self.shape,self.data_offsets,self.parameter_count,)==(other.dtype,other.shape,other.data_offsets,other.parameter_count,)
  return NotImplemented
 return __eq__