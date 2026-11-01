def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.device,self.dtype,self.inner_fn,self.ranges,self.scan_ranges,self.size,self.combine_fn,self.reindex,self.reduction_hint,self.output_index,self.dtypes,self.inner_fns,)==(other.device,other.dtype,other.inner_fn,other.ranges,other.scan_ranges,other.size,other.combine_fn,other.reindex,other.reduction_hint,other.output_index,other.dtypes,other.inner_fns,)
  return NotImplemented
 return __eq__