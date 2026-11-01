def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.tensor_state,self.zero_dim_tensor_state,self.non_tensor_state,)==(other.tensor_state,other.zero_dim_tensor_state,other.non_tensor_state,)
  return NotImplemented
 return __eq__