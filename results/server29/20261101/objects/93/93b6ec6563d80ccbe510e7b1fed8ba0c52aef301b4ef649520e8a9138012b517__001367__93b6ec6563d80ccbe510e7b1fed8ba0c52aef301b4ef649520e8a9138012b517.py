def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.base,self.member,)==(other.base,other.member,)
  return NotImplemented
 return __eq__