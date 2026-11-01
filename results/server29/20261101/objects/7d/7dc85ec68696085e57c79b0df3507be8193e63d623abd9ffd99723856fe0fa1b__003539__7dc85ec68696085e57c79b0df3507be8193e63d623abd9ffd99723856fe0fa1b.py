def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.time,self.perf_count,)==(other.time,other.perf_count,)
  return NotImplemented
 return __eq__