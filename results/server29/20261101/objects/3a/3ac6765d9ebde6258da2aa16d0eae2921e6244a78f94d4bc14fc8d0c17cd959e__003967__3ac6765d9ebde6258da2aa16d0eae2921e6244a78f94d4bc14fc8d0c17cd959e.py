def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.enabled,self.cache_enabled,)==(other.enabled,other.cache_enabled,)
  return NotImplemented
 return __eq__