def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.gradients_to_parameters,self.gradients_to_user_inputs,self.loss_output,)==(other.gradients_to_parameters,other.gradients_to_user_inputs,other.loss_output,)
  return NotImplemented
 return __eq__