def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.dynamic_sizes,self.dynamic_strides,self.constraint_sizes,self.constraint_strides,self.view_base_context,)==(other.dynamic_sizes,other.dynamic_strides,other.constraint_sizes,other.constraint_strides,other.view_base_context,)
  return NotImplemented
 return __eq__