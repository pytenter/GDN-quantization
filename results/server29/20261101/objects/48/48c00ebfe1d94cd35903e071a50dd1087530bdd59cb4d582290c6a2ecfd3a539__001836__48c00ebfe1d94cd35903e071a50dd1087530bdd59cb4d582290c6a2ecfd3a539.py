def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.module,self.param_name,self.shared_modules,self.shared_param_names,)==(other.module,other.param_name,other.shared_modules,other.shared_param_names,)
  return NotImplemented
 return __eq__