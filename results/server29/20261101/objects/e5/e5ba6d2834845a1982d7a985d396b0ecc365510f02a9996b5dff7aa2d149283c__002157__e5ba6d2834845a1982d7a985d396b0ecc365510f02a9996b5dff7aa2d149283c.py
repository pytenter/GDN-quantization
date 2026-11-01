def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.compile_pg,self.local_state,self.all_states,)==(other.compile_pg,other.local_state,other.all_states,)
  return NotImplemented
 return __eq__