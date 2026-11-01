def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.stage,self.hardware,self.requested_hardware,self.sleep_time,self.storage,self.raw,)==(other.stage,other.hardware,other.requested_hardware,other.sleep_time,other.storage,other.raw,)
  return NotImplemented
 return __eq__