def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self._expr,self.pytype,self._hint,self.constant,self.fx_node,)==(other._expr,other.pytype,other._hint,other.constant,other.fx_node,)
  return NotImplemented
 return __eq__