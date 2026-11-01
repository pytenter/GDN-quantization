def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.size_on_disk,self.repos,self.warnings,)==(other.size_on_disk,other.repos,other.warnings,)
  return NotImplemented
 return __eq__