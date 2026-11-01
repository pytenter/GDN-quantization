def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.as_none,self.as_int,self.as_float,self.as_string,self.as_bool,)==(other.as_none,other.as_int,other.as_float,other.as_string,other.as_bool,)
  return NotImplemented
 return __eq__