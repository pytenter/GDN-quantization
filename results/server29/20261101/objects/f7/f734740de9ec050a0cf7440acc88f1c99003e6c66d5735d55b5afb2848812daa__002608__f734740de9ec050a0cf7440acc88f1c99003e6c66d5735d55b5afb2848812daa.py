def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.backend,self.subsystem,self.bisect_number,self.debug_info,)==(other.backend,other.subsystem,other.bisect_number,other.debug_info,)
  return NotImplemented
 return __eq__