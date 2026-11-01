def __create_fn__():
 def __hash__(self):
  return hash((self.dtype,self.layout,self.device,self.pin_memory,))
 return __hash__