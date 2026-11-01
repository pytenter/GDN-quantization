def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.value,self.error,)==(other.value,other.error,)
  return NotImplemented
 return __eq__