def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.name,self.mutating_buf,)==(other.name,other.mutating_buf,)
  return NotImplemented
 return __eq__