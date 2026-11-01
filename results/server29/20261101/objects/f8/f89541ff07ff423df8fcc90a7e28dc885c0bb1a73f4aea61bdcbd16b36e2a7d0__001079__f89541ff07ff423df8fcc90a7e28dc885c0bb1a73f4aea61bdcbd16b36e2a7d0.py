def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.default,self.value_type,self.user_override,self.justknob,self.env_value_force,self.env_value_default,self.hide,self.alias,)==(other.default,other.value_type,other.user_override,other.justknob,other.env_value_force,other.env_value_default,other.hide,other.alias,)
  return NotImplemented
 return __eq__