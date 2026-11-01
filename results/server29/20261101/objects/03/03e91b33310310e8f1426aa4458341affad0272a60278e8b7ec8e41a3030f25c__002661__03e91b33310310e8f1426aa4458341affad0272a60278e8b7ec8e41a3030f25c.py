def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.as_none,self.as_tensor,self.as_tensors,self.as_int,self.as_ints,self.as_float,self.as_floats,self.as_string,self.as_strings,self.as_sym_int,self.as_sym_ints,self.as_scalar_type,self.as_memory_format,self.as_layout,self.as_device,self.as_bool,self.as_bools,self.as_sym_bool,self.as_sym_bools,self.as_graph,self.as_optional_tensors,self.as_custom_obj,self.as_operator,self.as_sym_float,self.as_sym_floats,)==(other.as_none,other.as_tensor,other.as_tensors,other.as_int,other.as_ints,other.as_float,other.as_floats,other.as_string,other.as_strings,other.as_sym_int,other.as_sym_ints,other.as_scalar_type,other.as_memory_format,other.as_layout,other.as_device,other.as_bool,other.as_bools,other.as_sym_bool,other.as_sym_bools,other.as_graph,other.as_optional_tensors,other.as_custom_obj,other.as_operator,other.as_sym_float,other.as_sym_floats,)
  return NotImplemented
 return __eq__