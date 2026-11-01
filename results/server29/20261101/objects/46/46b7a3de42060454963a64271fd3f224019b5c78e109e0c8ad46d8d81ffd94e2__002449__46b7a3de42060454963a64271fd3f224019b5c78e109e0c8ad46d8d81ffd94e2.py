def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.unwrapped_idx,self.memory_format,)==(other.unwrapped_idx,other.memory_format,)
  return NotImplemented
 return __eq__