def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.qconfig_mapping,self.example_inputs,self.prepare_custom_config,self.backend_config,)==(other.qconfig_mapping,other.example_inputs,other.prepare_custom_config,other.backend_config,)
  return NotImplemented
 return __eq__