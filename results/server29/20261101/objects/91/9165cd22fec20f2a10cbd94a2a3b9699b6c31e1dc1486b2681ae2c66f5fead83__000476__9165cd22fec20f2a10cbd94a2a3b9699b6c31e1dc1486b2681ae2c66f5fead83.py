def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.id,self.size,self.data,)==(other.id,other.size,other.data,)
  return NotImplemented
 return __eq__