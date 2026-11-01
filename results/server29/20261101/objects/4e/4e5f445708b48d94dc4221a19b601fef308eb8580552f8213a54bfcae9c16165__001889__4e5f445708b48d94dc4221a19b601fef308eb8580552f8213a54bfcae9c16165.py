def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.cleanup_fn,self.proxy,)==(other.cleanup_fn,other.proxy,)
  return NotImplemented
 return __eq__