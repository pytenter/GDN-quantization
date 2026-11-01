def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.id,self.at,)==(other.id,other.at,)
  return NotImplemented
 return __eq__