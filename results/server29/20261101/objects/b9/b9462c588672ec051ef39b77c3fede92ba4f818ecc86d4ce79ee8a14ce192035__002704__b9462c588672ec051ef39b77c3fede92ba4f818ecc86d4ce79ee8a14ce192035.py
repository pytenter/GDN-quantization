def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.arg,self.buffer_name,)==(other.arg,other.buffer_name,)
  return NotImplemented
 return __eq__