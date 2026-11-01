def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.default,self.justknob,self.env_name_default,self.env_name_force,self.alias,)==(other.default,other.justknob,other.env_name_default,other.env_name_force,other.alias,)
  return NotImplemented
 return __eq__