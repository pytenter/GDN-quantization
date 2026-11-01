def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.device,self.dtype,self.inner_fn,self.ranges,self.output_indexer,self.scatter_mode,)==(other.device,other.dtype,other.inner_fn,other.ranges,other.output_indexer,other.scatter_mode,)
  return NotImplemented
 return __eq__