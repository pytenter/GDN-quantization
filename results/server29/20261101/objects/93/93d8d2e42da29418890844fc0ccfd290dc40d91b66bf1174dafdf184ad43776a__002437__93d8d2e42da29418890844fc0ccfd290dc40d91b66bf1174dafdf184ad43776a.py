def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.output_type,self.raw_type,self.base_idx,self.dynamic_dims,self.requires_grad,self.functional_tensor,)==(other.output_type,other.raw_type,other.base_idx,other.dynamic_dims,other.requires_grad,other.functional_tensor,)
  return NotImplemented
 return __eq__