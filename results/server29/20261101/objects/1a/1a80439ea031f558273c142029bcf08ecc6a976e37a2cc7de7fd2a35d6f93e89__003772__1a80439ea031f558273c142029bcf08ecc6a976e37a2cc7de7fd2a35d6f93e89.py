def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.sha256,self.size,self.sample,)==(other.sha256,other.size,other.sample,)
  return NotImplemented
 return __eq__