def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.last_hidden_state,self.extract_features,self.hidden_states,self.attentions,)==(other.last_hidden_state,other.extract_features,other.hidden_states,other.attentions,)
  return NotImplemented
 return __eq__