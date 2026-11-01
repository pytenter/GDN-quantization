def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.expr,self.msg,self.stack,)==(other.expr,other.msg,other.stack,)
  return NotImplemented
 return __eq__