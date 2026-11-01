def __create_fn__():
 def __hash__(self):
  return hash((self.grad_input,self.captured_grads_compute,self.captured_grads,self.mutated_grads,))
 return __hash__