def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.device,self.dtype,self.inner_fn,self.ranges,self.reduction_ranges,self.reduction_type,self.src_dtype,self.reduction_hint,)==(other.device,other.dtype,other.inner_fn,other.ranges,other.reduction_ranges,other.reduction_type,other.src_dtype,other.reduction_hint,)
  return NotImplemented
 return __eq__