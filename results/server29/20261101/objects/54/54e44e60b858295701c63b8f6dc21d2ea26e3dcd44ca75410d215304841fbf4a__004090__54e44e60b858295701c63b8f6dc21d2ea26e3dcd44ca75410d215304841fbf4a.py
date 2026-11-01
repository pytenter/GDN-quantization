def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.last_hidden_state,self.pooler_output,self.hidden_states,)==(other.last_hidden_state,other.pooler_output,other.hidden_states,)
  return NotImplemented
 return __eq__