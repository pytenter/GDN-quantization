def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.level,self.mode,self.write_thread_id,self.marker_name,self.is_reader,self.token,self.heartbeat_thread,self.heartbeat_stop,)==(other.level,other.mode,other.write_thread_id,other.marker_name,other.is_reader,other.token,other.heartbeat_thread,other.heartbeat_stop,)
  return NotImplemented
 return __eq__