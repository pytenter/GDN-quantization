def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.output_specs,self.input_specs,self.redistribute_cost,)==(other.output_specs,other.input_specs,other.redistribute_cost,)
  return NotImplemented
 return __eq__