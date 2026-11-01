def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.action,self.allow_empty,self.was_matched,)==(other.action,other.allow_empty,other.was_matched,)
  return NotImplemented
 return __eq__