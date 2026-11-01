def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.state,self.handle,self.param_indices,self.param_requires_grad,)==(other.state,other.handle,other.param_indices,other.param_requires_grad,)
  return NotImplemented
 return __eq__