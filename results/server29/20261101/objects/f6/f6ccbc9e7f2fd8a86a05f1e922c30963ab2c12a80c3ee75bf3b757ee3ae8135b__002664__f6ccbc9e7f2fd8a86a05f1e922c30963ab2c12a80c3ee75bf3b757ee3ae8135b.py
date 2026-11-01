def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.name,self.arg,self.kind,)==(other.name,other.arg,other.kind,)
  return NotImplemented
 return __eq__