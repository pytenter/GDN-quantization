def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.parameters,self.buffers,self.user_inputs,self.user_outputs,self.inputs_to_parameters,self.inputs_to_buffers,self.buffers_to_mutate,self.user_inputs_to_mutate,self.in_spec,self.out_spec,self.backward_signature,self.input_tokens,self.output_tokens,)==(other.parameters,other.buffers,other.user_inputs,other.user_outputs,other.inputs_to_parameters,other.inputs_to_buffers,other.buffers_to_mutate,other.user_inputs_to_mutate,other.in_spec,other.out_spec,other.backward_signature,other.input_tokens,other.output_tokens,)
  return NotImplemented
 return __eq__