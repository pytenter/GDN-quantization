def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.input_dtype_with_constraints,self.output_dtype_with_constraints,self.weight_dtype_with_constraints,self.bias_dtype,self.is_dynamic,)==(other.input_dtype_with_constraints,other.output_dtype_with_constraints,other.weight_dtype_with_constraints,other.bias_dtype,other.is_dynamic,)
  return NotImplemented
 return __eq__