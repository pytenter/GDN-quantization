def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.miss,self.hit,self.put,self.exception,)==(other.miss,other.hit,other.put,other.exception,)
  return NotImplemented
 return __eq__