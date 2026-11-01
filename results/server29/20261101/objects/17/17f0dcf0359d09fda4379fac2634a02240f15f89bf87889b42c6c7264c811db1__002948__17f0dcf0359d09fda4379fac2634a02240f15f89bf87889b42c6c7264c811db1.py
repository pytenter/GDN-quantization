def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.data,self.target_dtype,)==(other.data,other.target_dtype,)
  return NotImplemented
 return __eq__