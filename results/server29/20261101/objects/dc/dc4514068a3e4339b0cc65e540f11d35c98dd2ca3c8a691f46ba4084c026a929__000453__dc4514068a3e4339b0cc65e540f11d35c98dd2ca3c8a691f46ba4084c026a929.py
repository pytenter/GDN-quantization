def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.overlapping_sources,self.non_overlapping_sources,)==(other.overlapping_sources,other.non_overlapping_sources,)
  return NotImplemented
 return __eq__