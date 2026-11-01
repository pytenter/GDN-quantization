def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.inputs,self.outputs,self.nodes,self.tensor_values,self.sym_int_values,self.sym_bool_values,self.is_single_tensor_return,self.custom_obj_values,self.sym_float_values,)==(other.inputs,other.outputs,other.nodes,other.tensor_values,other.sym_int_values,other.sym_bool_values,other.is_single_tensor_return,other.custom_obj_values,other.sym_float_values,)
  return NotImplemented
 return __eq__