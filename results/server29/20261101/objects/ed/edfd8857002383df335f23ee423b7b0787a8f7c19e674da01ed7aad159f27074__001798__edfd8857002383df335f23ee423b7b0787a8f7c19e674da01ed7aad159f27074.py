def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.input_dim,self.times,)==(other.input_dim,other.times,)
  return NotImplemented
 return __eq__