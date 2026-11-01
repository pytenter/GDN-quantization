def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.enable_expert_parallel,)==(other.enable_expert_parallel,)
  return NotImplemented
 return __eq__