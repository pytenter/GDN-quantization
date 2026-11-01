def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.indices_of_inps_to_detach,self.trace_joint,self.disable_amp,)==(other.indices_of_inps_to_detach,other.trace_joint,other.disable_amp,)
  return NotImplemented
 return __eq__