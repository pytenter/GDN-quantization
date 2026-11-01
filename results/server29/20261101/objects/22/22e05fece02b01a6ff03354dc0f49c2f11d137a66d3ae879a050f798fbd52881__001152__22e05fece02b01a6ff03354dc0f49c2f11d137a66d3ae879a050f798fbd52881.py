def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.total_tt,self.func_profiles,)==(other.total_tt,other.func_profiles,)
  return NotImplemented
 return __eq__