def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.as_name,self.as_bool,)==(other.as_name,other.as_bool,)
  return NotImplemented
 return __eq__