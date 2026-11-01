def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.internal,self.transaction,self.state,)==(other.internal,other.transaction,other.state,)
  return NotImplemented
 return __eq__