def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.expr_str,self.hint,)==(other.expr_str,other.hint,)
  return NotImplemented
 return __eq__