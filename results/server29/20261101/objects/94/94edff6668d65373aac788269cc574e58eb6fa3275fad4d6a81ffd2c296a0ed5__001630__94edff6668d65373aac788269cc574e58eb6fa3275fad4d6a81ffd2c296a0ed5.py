def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.inductor_artifacts,self.autotune_artifacts,self.aot_autograd_artifacts,self.pgo_artifacts,)==(other.inductor_artifacts,other.autotune_artifacts,other.aot_autograd_artifacts,other.pgo_artifacts,)
  return NotImplemented
 return __eq__