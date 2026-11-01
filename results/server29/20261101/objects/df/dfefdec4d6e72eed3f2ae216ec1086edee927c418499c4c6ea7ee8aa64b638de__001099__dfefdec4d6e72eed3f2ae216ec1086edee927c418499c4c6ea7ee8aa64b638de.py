def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.backend,self.arch,self.warp_size,)==(other.backend,other.arch,other.warp_size,)
  return NotImplemented
 return __eq__