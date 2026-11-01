def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.frames,self.message,self.properties,)==(other.frames,other.message,other.properties,)
  return NotImplemented
 return __eq__