def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.user_output,self.loss_output,self.buffer_mutation,self.gradient_to_parameter,self.gradient_to_user_input,self.user_input_mutation,self.token,)==(other.user_output,other.loss_output,other.buffer_mutation,other.gradient_to_parameter,other.gradient_to_user_input,other.user_input_mutation,other.token,)
  return NotImplemented
 return __eq__