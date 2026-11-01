def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.name,self.buffer,self.dtype,self.offset,self.alias_of,)==(other.name,other.buffer,other.dtype,other.offset,other.alias_of,)
  return NotImplemented
 return __eq__