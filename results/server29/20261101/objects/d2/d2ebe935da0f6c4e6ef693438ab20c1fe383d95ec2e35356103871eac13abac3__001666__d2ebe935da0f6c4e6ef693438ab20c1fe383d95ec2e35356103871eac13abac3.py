def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.offload_to_cpu,self._use_dtensor,)==(other.offload_to_cpu,other._use_dtensor,)
  return NotImplemented
 return __eq__