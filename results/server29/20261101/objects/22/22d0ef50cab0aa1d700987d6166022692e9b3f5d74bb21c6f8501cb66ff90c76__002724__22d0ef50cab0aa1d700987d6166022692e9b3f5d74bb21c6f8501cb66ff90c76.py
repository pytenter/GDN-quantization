def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.min_val,self.max_val,)==(other.min_val,other.max_val,)
  return NotImplemented
 return __eq__