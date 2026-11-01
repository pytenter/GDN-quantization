def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.dim,self.loop,self.bytes_contiguous_or_broadcast,self.bytes_non_contiguous,)==(other.dim,other.loop,other.bytes_contiguous_or_broadcast,other.bytes_non_contiguous,)
  return NotImplemented
 return __eq__