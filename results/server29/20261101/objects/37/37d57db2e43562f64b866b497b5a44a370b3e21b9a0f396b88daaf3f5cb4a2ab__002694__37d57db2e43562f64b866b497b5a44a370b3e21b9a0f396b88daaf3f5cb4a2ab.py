def __create_fn__(_type_user_input, _type_parameter, _type_buffer, _type_tensor_constant, _type_custom_obj, _type_token, _type_constant_input, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _return_type):
 def __init__(self,user_input:_type_user_input,parameter:_type_parameter,buffer:_type_buffer,tensor_constant:_type_tensor_constant,custom_obj:_type_custom_obj,token:_type_token,constant_input:_type_constant_input)->_return_type:
  self.user_input=user_input
  self.parameter=parameter
  self.buffer=buffer
  self.tensor_constant=tensor_constant
  self.custom_obj=custom_obj
  self.token=token
  self.constant_input=constant_input
  self.__post_init__()
 return __init__