def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.name,self.args,self.kwargs,self._param_ids_from,self._param_ids_generated,)==(other.name,other.args,other.kwargs,other._param_ids_from,other._param_ids_generated,)
  return NotImplemented
 return __eq__