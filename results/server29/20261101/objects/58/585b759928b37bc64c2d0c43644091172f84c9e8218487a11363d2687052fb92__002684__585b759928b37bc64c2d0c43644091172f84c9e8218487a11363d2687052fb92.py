def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.arg,self.buffer_name,self.persistent,)==(other.arg,other.buffer_name,other.persistent,)
  return NotImplemented
 return __eq__