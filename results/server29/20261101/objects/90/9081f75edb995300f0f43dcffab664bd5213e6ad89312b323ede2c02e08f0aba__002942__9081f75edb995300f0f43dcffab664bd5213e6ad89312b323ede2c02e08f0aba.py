def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.data,self.layout,)==(other.data,other.layout,)
  return NotImplemented
 return __eq__