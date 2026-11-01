def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.cache_key,self.start_time_ns,)==(other.cache_key,other.start_time_ns,)
  return NotImplemented
 return __eq__