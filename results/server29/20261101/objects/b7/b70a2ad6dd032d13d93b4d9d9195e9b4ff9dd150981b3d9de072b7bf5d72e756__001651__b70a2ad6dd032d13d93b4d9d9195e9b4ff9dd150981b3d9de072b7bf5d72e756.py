def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.param_dtype,self.reduce_dtype,self.buffer_dtype,self.keep_low_precision_grads,self.cast_forward_inputs,self.cast_root_forward_inputs,self._module_classes_to_ignore,)==(other.param_dtype,other.reduce_dtype,other.buffer_dtype,other.keep_low_precision_grads,other.cast_forward_inputs,other.cast_root_forward_inputs,other._module_classes_to_ignore,)
  return NotImplemented
 return __eq__