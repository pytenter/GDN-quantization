def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.library_basename,self.input_names,self.output_names,self.floating_point_input_dtype,self.floating_point_output_dtype,self.aot_inductor_model_is_cpu,)==(other.library_basename,other.input_names,other.output_names,other.floating_point_input_dtype,other.floating_point_output_dtype,other.aot_inductor_model_is_cpu,)
  return NotImplemented
 return __eq__