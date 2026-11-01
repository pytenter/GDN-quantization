def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.arg,self.parameter_name,)==(other.arg,other.parameter_name,)
  return NotImplemented
 return __eq__