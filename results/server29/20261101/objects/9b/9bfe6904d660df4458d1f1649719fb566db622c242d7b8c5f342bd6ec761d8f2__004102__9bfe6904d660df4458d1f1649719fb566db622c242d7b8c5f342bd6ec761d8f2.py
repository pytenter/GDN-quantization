def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.last_hidden_state,self.past_key_values,self.hidden_states,self.attentions,self.cross_attentions,)==(other.last_hidden_state,other.past_key_values,other.hidden_states,other.attentions,other.cross_attentions,)
  return NotImplemented
 return __eq__