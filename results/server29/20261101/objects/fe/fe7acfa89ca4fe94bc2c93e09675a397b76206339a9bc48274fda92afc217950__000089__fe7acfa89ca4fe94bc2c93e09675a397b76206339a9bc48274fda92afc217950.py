def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.dispatch_key,self.use_out_as_primary,self.device_guard,self.external,self.index,)==(other.dispatch_key,other.use_out_as_primary,other.device_guard,other.external,other.index,)
  return NotImplemented
 return __eq__