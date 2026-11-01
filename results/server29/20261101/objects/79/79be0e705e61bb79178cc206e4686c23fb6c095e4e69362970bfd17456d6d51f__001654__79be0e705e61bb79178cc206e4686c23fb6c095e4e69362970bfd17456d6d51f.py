def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.offload_params,)==(other.offload_params,)
  return NotImplemented
 return __eq__