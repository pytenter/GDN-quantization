def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.name,self.is_constexpr,)==(other.name,other.is_constexpr,)
  return NotImplemented
 return __eq__