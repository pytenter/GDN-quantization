def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.cached_kernel_names,)==(other.cached_kernel_names,)
  return NotImplemented
 return __eq__