def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.pattern,self.extra_check,self.handler,)==(other.pattern,other.extra_check,other.handler,)
  return NotImplemented
 return __eq__