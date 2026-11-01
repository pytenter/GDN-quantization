def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.as_int,self.as_bool,self.as_float,)==(other.as_int,other.as_bool,other.as_float,)
  return NotImplemented
 return __eq__