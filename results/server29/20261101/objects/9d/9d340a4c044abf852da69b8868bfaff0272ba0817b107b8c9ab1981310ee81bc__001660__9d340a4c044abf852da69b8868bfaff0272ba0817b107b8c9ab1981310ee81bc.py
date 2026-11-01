def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.offload_to_cpu,self.rank0_only,)==(other.offload_to_cpu,other.rank0_only,)
  return NotImplemented
 return __eq__