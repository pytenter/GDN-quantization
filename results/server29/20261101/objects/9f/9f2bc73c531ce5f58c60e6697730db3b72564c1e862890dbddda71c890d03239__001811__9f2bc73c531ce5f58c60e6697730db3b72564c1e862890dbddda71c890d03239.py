def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.param_dtype,self.reduce_dtype,self.output_dtype,self.cast_forward_inputs,)==(other.param_dtype,other.reduce_dtype,other.output_dtype,other.cast_forward_inputs,)
  return NotImplemented
 return __eq__