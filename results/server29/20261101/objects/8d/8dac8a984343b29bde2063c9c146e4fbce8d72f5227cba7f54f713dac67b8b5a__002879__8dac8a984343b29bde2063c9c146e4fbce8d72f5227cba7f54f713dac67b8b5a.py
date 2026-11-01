def __create_fn__():
 def __hash__(self):
  return hash((self.device,self.dtype,self.inner_fn,self.ranges,self.output_indexer,self.scatter_mode,))
 return __hash__