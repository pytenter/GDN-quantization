def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.params,self.indices,self._arg2scope,self._idlist,self.marks,)==(other.params,other.indices,other._arg2scope,other._idlist,other.marks,)
  return NotImplemented
 return __eq__