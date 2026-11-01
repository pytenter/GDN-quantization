def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.func,self.location,)==(other.func,other.location,)
  return NotImplemented
 return __eq__