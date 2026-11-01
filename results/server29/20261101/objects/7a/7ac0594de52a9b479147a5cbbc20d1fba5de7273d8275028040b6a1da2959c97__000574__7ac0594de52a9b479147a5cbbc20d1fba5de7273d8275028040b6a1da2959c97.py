def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.hits,self.misses,self.bypasses,self.size,)==(other.hits,other.misses,other.bypasses,other.size,)
  return NotImplemented
 return __eq__