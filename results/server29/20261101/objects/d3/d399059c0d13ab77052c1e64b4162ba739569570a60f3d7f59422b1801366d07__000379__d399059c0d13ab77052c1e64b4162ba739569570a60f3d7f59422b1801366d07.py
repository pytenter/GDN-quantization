def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.input_version,self.mutated,)==(other.input_version,other.mutated,)
  return NotImplemented
 return __eq__