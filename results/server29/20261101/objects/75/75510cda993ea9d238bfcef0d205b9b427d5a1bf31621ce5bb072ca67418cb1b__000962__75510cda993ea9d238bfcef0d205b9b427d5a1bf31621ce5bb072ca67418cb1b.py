def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.input_spec,self.output_spec,self.result_is_tuple,)==(other.input_spec,other.output_spec,other.result_is_tuple,)
  return NotImplemented
 return __eq__