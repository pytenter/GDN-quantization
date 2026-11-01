def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.sequences,self.sequences_scores,self.scores,self.logits,self.beam_indices,self.attentions,self.hidden_states,self.past_key_values,)==(other.sequences,other.sequences_scores,other.scores,other.logits,other.beam_indices,other.attentions,other.hidden_states,other.past_key_values,)
  return NotImplemented
 return __eq__