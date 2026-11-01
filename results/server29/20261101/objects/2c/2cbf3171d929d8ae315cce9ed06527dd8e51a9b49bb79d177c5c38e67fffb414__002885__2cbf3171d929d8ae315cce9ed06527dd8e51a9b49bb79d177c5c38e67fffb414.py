def __create_fn__():
 def __hash__(self):
  return hash((self.device,self.dtype,self.inner_fn,self.ranges,self.reduction_ranges,self.reduction_type,self.src_dtype,self.reduction_hint,))
 return __hash__