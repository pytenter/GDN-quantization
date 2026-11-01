def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.grad_input,self.captured_grads_compute,self.captured_grads,self.mutated_grads,)==(other.grad_input,other.captured_grads_compute,other.captured_grads,other.mutated_grads,)
  return NotImplemented
 return __eq__