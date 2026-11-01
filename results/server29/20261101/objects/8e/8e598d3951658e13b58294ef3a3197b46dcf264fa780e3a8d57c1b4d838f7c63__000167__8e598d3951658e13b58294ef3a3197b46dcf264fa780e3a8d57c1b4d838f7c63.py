def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.base,self.inplace,self.dunder_method,self.functional_overload,self.namespace,)==(other.base,other.inplace,other.dunder_method,other.functional_overload,other.namespace,)
  return NotImplemented
 return __eq__