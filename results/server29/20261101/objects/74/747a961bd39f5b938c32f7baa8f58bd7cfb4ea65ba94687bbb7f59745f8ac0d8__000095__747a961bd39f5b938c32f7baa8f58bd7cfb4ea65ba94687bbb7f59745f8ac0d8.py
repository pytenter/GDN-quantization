def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.name,self.arguments,self.returns,)==(other.name,other.arguments,other.returns,)
  return NotImplemented
 return __eq__