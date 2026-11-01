def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.data,self.refcount,)==(other.data,other.refcount,)
  return NotImplemented
 return __eq__