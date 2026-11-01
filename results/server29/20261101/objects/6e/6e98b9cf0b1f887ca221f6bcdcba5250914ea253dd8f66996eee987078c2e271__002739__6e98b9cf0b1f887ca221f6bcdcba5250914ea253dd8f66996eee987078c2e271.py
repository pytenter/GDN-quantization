def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.major,self.minor,)==(other.major,other.minor,)
  return NotImplemented
 return __eq__