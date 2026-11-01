def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.lower,self.upper,self.is_bool,self.is_int,self.is_float,)==(other.lower,other.upper,other.is_bool,other.is_int,other.is_float,)
  return NotImplemented
 return __eq__