def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.as_expr,self.as_int,)==(other.as_expr,other.as_int,)
  return NotImplemented
 return __eq__