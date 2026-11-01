def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.inner_kernel_number,self.local_buffer_number,)==(other.inner_kernel_number,other.local_buffer_number,)
  return NotImplemented
 return __eq__