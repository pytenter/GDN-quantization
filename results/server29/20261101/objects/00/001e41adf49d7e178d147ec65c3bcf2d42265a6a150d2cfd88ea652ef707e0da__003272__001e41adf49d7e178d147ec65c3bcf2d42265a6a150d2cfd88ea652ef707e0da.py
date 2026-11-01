def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.parallel_depth,self.start_depth,)==(other.parallel_depth,other.start_depth,)
  return NotImplemented
 return __eq__