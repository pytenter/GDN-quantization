def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.name,self.type,self.default,self.annotation,)==(other.name,other.type,other.default,other.annotation,)
  return NotImplemented
 return __eq__