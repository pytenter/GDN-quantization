def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.size,self.sha256,self.pointer_size,)==(other.size,other.sha256,other.pointer_size,)
  return NotImplemented
 return __eq__