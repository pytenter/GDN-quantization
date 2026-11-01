def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.sequences,self.scores,self.logits,self.attentions,self.hidden_states,self.past_key_values,)==(other.sequences,other.scores,other.logits,other.attentions,other.hidden_states,other.past_key_values,)
  return NotImplemented
 return __eq__