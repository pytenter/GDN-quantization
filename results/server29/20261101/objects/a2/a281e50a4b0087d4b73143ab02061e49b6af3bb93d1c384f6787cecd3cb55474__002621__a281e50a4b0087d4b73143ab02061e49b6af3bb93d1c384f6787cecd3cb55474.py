def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.start_idx,self.num_tensors,self.class_type,self.attrs,self.metadata,self.outer_size,self.outer_stride,)==(other.start_idx,other.num_tensors,other.class_type,other.attrs,other.metadata,other.outer_size,other.outer_stride,)
  return NotImplemented
 return __eq__