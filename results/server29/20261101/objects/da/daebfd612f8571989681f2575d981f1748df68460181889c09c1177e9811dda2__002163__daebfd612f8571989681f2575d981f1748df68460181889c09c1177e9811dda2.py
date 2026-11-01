def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self._exc_stack,self._current_exception,)==(other._exc_stack,other._current_exception,)
  return NotImplemented
 return __eq__