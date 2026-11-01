def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.params,self.constant_offset,self.order,self.mask_vars,self.broadcast_shape,self.broadcasting_dims,self.final_shape,self._boundary_check,)==(other.params,other.constant_offset,other.order,other.mask_vars,other.broadcast_shape,other.broadcasting_dims,other.final_shape,other._boundary_check,)
  return NotImplemented
 return __eq__