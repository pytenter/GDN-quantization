def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.input_source_a,self.input_source_b,)==(other.input_source_a,other.input_source_b,)
  return NotImplemented
 return __eq__