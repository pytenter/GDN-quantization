def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.state,self.write,self.readers,)==(other.state,other.write,other.readers,)
  return NotImplemented
 return __eq__