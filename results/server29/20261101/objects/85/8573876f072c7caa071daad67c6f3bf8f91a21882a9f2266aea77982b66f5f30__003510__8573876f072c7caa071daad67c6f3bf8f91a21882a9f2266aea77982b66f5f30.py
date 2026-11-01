def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.type,self.value,self.pos,)==(other.type,other.value,other.pos,)
  return NotImplemented
 return __eq__