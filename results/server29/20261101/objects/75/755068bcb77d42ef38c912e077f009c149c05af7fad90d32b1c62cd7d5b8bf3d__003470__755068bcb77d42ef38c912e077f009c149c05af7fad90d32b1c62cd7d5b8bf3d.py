def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self._excinfo,self._striptext,self._traceback,)==(other._excinfo,other._striptext,other._traceback,)
  return NotImplemented
 return __eq__