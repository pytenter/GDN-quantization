def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.state_dict_type,self.state_dict_config,self.optim_state_dict_config,)==(other.state_dict_type,other.state_dict_config,other.optim_state_dict_config,)
  return NotImplemented
 return __eq__