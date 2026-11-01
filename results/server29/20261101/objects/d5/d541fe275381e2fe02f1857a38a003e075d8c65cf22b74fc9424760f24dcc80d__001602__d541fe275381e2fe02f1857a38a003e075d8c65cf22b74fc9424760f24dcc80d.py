def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.expr_count,self.expr_to_name,)==(other.expr_count,other.expr_to_name,)
  return NotImplemented
 return __eq__