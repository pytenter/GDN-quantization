def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.proxy,self.constant,)==(other.proxy,other.constant,)
  return NotImplemented
 return __eq__