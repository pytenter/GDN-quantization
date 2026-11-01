def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.contracting_dims,self.batch_dims,self.lhs_out_only_dims,self.rhs_out_only_dims,)==(other.contracting_dims,other.batch_dims,other.lhs_out_only_dims,other.rhs_out_only_dims,)
  return NotImplemented
 return __eq__