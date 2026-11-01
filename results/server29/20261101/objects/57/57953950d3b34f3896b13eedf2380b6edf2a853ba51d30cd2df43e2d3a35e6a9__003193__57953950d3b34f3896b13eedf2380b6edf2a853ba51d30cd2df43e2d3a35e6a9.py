def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.category,self.key,self.self_device_time_ms,self.count,)==(other.category,other.key,other.self_device_time_ms,other.count,)
  return NotImplemented
 return __eq__