def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.location,self.state,self.index,self.stack,)==(other.location,other.state,other.index,other.stack,)
  return NotImplemented
 return __eq__