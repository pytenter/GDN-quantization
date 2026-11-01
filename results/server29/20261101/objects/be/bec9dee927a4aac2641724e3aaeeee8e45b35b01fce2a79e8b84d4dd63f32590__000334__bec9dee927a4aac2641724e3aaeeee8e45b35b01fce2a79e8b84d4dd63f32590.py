def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.file,self.lineno,self.name,self.code,)==(other.file,other.lineno,other.name,other.code,)
  return NotImplemented
 return __eq__