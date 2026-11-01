def __create_fn__():
 def __hash__(self):
  return hash((self.device,self.dtype,self.inner_fn,self.ranges,))
 return __hash__