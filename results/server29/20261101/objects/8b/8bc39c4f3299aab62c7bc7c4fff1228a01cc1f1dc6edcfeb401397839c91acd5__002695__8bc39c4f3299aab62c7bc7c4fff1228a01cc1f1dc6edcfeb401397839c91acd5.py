def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.user_input,self.parameter,self.buffer,self.tensor_constant,self.custom_obj,self.token,self.constant_input,)==(other.user_input,other.parameter,other.buffer,other.tensor_constant,other.custom_obj,other.token,other.constant_input,)
  return NotImplemented
 return __eq__