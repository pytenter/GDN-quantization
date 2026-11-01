def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.count,self.zero_mode,self.device,self.outer_name,self.inner_name,self.dtype,)==(other.count,other.zero_mode,other.device,other.outer_name,other.inner_name,other.dtype,)
  return NotImplemented
 return __eq__