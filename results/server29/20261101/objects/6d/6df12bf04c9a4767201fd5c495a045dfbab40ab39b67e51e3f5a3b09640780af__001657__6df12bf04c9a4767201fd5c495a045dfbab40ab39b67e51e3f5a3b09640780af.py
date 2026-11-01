def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.offload_to_cpu,)==(other.offload_to_cpu,)
  return NotImplemented
 return __eq__