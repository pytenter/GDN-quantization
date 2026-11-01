def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.functional,self.inplace,self.mutable,self.out,)==(other.functional,other.inplace,other.mutable,other.out,)
  return NotImplemented
 return __eq__