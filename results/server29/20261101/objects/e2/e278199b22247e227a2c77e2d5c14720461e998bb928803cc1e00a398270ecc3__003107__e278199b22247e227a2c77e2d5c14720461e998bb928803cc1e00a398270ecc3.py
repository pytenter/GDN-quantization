def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.dtype,self.quant_min_lower_bound,self.quant_max_upper_bound,self.scale_min_lower_bound,self.scale_max_upper_bound,self.scale_exact_match,self.zero_point_exact_match,)==(other.dtype,other.quant_min_lower_bound,other.quant_max_upper_bound,other.scale_min_lower_bound,other.scale_max_upper_bound,other.scale_exact_match,other.zero_point_exact_match,)
  return NotImplemented
 return __eq__