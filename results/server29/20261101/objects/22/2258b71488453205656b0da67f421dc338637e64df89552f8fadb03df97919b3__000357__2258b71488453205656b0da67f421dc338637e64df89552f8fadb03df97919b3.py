def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.duration_time_ns,self.self_time_ns,self.idle_time_ns,self.queue_depth,)==(other.duration_time_ns,other.self_time_ns,other.idle_time_ns,other.queue_depth,)
  return NotImplemented
 return __eq__