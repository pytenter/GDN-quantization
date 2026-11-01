def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self._backward_fn,self._setup_context_fn,)==(other._backward_fn,other._setup_context_fn,)
  return NotImplemented
 return __eq__