def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.triton,self.extern,self.expected,)==(other.triton,other.extern,other.expected,)
  return NotImplemented
 return __eq__