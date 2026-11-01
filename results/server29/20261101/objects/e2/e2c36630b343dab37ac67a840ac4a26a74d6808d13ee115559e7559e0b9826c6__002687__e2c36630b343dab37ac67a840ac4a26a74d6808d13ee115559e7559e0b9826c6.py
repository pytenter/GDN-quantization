def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.arg,self.tensor_constant_name,)==(other.arg,other.tensor_constant_name,)
  return NotImplemented
 return __eq__