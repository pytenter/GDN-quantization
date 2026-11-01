def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.input_specs,self.output_specs,)==(other.input_specs,other.output_specs,)
  return NotImplemented
 return __eq__