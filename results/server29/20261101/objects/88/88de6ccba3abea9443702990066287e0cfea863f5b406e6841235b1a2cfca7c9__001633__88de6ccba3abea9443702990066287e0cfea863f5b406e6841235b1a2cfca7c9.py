def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.filename,self.firstlineno,self.name,)==(other.filename,other.firstlineno,other.name,)
  return NotImplemented
 return __eq__