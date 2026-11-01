def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self._given_basetemp,self._trace,self._basetemp,self._retention_count,self._retention_policy,)==(other._given_basetemp,other._trace,other._basetemp,other._retention_count,other._retention_policy,)
  return NotImplemented
 return __eq__