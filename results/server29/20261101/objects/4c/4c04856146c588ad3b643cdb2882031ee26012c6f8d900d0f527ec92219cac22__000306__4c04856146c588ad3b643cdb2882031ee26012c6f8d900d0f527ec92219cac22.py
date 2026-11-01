def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.buffer_comm_hook,self.buffer_comm_hook_state,self.buffer_comm_hook_location,)==(other.buffer_comm_hook,other.buffer_comm_hook_state,other.buffer_comm_hook_location,)
  return NotImplemented
 return __eq__