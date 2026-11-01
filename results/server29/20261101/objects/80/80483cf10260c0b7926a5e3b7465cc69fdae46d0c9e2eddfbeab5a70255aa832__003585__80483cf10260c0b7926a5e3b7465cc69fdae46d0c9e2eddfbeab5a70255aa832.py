def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self._cachedir,self._config,)==(other._cachedir,other._config,)
  return NotImplemented
 return __eq__