def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.kernel_hash,self.device,self.artifacts,)==(other.kernel_hash,other.device,other.artifacts,)
  return NotImplemented
 return __eq__