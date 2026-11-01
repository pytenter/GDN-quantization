def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.pattern,self.extra_check,)==(other.pattern,other.extra_check,)
  return NotImplemented
 return __eq__