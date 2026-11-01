def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.pre_self_positional,self.self_arg,self.post_self_positional,self.pre_tensor_options_kwarg_only,self.tensor_options,self.post_tensor_options_kwarg_only,self.out,)==(other.pre_self_positional,other.self_arg,other.post_self_positional,other.pre_tensor_options_kwarg_only,other.tensor_options,other.post_tensor_options_kwarg_only,other.out,)
  return NotImplemented
 return __eq__