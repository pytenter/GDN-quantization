def __create_fn__(_type_parameters, _type_buffers, _type_user_inputs, _type_user_outputs, _type_inputs_to_parameters, _type_inputs_to_buffers, _type_buffers_to_mutate, _type_user_inputs_to_mutate, _type_in_spec, _type_out_spec, _type_backward_signature, _type_input_tokens, _type_output_tokens, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _return_type):
 def __init__(self,parameters:_type_parameters,buffers:_type_buffers,user_inputs:_type_user_inputs,user_outputs:_type_user_outputs,inputs_to_parameters:_type_inputs_to_parameters,inputs_to_buffers:_type_inputs_to_buffers,buffers_to_mutate:_type_buffers_to_mutate,user_inputs_to_mutate:_type_user_inputs_to_mutate,in_spec:_type_in_spec,out_spec:_type_out_spec,backward_signature:_type_backward_signature,input_tokens:_type_input_tokens,output_tokens:_type_output_tokens)->_return_type:
  self.parameters=parameters
  self.buffers=buffers
  self.user_inputs=user_inputs
  self.user_outputs=user_outputs
  self.inputs_to_parameters=inputs_to_parameters
  self.inputs_to_buffers=inputs_to_buffers
  self.buffers_to_mutate=buffers_to_mutate
  self.user_inputs_to_mutate=user_inputs_to_mutate
  self.in_spec=in_spec
  self.out_spec=out_spec
  self.backward_signature=backward_signature
  self.input_tokens=input_tokens
  self.output_tokens=output_tokens
 return __init__