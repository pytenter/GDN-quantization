def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self._current_time,)==(other._current_time,)
  return NotImplemented
 return __eq__