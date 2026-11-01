def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.inner_exceptions,self.kind,self.message,self.properties,self.stack,)==(other.inner_exceptions,other.kind,other.message,other.properties,other.stack,)
  return NotImplemented
 return __eq__