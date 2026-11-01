def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.name,self.config_name,self.config_field,self.config_value,)==(other.name,other.config_name,other.config_field,other.config_value,)
  return NotImplemented
 return __eq__