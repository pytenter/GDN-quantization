def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.name,self.stack_trace,self.users,self.mutating_use_stack_trace,)==(other.name,other.stack_trace,other.users,other.mutating_use_stack_trace,)
  return NotImplemented
 return __eq__