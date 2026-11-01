def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.name,self.layout,self.data,)==(other.name,other.layout,other.data,)
  return NotImplemented
 return __eq__