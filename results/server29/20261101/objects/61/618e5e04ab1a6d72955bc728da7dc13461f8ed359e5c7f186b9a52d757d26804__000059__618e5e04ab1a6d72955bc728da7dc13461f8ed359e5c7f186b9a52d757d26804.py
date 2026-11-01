def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.file,self.line,)==(other.file,other.line,)
  return NotImplemented
 return __eq__