def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.flat_tensor_start_idx,self.arg_count,self.included_subclass_symints,self.attrs,self.outer_size,self.outer_stride,self.meta,self.original_subclass,self.original_subclass_type,self.memory_format,)==(other.flat_tensor_start_idx,other.arg_count,other.included_subclass_symints,other.attrs,other.outer_size,other.outer_stride,other.meta,other.original_subclass,other.original_subclass_type,other.memory_format,)
  return NotImplemented
 return __eq__