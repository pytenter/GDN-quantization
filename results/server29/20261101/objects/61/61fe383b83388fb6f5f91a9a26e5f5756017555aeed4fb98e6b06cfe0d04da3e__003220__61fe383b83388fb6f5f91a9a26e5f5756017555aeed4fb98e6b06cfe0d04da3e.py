def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.device_idx,self.last_seen_device_guard_index,)==(other.device_idx,other.last_seen_device_guard_index,)
  return NotImplemented
 return __eq__