def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.inner,self.inner_expr,)==(other.inner,other.inner_expr,)
  return NotImplemented
 return __eq__