def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.init_scale,self.growth_factor,self.backoff_factor,self.growth_interval,self.enabled,)==(other.init_scale,other.growth_factor,other.backoff_factor,other.growth_interval,other.enabled,)
  return NotImplemented
 return __eq__