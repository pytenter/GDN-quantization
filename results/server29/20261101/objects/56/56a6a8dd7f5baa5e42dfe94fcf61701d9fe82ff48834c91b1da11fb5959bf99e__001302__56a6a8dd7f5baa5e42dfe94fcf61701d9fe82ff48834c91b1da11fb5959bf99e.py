def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.low,self.high,self.fixed_point,)==(other.low,other.high,other.fixed_point,)
  return NotImplemented
 return __eq__