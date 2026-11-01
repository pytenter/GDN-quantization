def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.input_dtype,self.input2_dtype,self.output_dtype,self.compute_dtype,self.vec_isa_cls,self.register_blocking,self.extra_check,)==(other.input_dtype,other.input2_dtype,other.output_dtype,other.compute_dtype,other.vec_isa_cls,other.register_blocking,other.extra_check,)
  return NotImplemented
 return __eq__