def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.kernel_name,self.triton_version,self.autotune_entries,self.default_config,)==(other.kernel_name,other.triton_version,other.autotune_entries,other.default_config,)
  return NotImplemented
 return __eq__