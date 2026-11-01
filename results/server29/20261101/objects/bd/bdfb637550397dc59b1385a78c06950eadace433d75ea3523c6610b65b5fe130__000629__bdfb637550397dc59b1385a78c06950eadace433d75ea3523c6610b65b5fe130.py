def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.kind,self.arg,self.target,)==(other.kind,other.arg,other.target,)
  return NotImplemented
 return __eq__