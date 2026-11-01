def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.flat_requires_grad,)==(other.flat_requires_grad,)
  return NotImplemented
 return __eq__