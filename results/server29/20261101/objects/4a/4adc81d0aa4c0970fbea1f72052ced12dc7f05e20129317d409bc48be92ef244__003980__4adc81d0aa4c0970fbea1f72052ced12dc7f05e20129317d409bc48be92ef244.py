def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.config,self.module_filter_func,self.pad_inner_dim,self.enable_fsdp_float8_all_gather,)==(other.config,other.module_filter_func,other.pad_inner_dim,other.enable_fsdp_float8_all_gather,)
  return NotImplemented
 return __eq__