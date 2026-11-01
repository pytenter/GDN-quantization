def __create_fn__(_type_user_output, _type_loss_output, _type_buffer_mutation, _type_gradient_to_parameter, _type_gradient_to_user_input, _type_user_input_mutation, _type_token, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _return_type):
 def __init__(self,user_output:_type_user_output,loss_output:_type_loss_output,buffer_mutation:_type_buffer_mutation,gradient_to_parameter:_type_gradient_to_parameter,gradient_to_user_input:_type_gradient_to_user_input,user_input_mutation:_type_user_input_mutation,token:_type_token)->_return_type:
  self.user_output=user_output
  self.loss_output=loss_output
  self.buffer_mutation=buffer_mutation
  self.gradient_to_parameter=gradient_to_parameter
  self.gradient_to_user_input=gradient_to_user_input
  self.user_input_mutation=user_input_mutation
  self.token=token
  self.__post_init__()
 return __init__