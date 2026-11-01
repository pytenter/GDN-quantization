def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.sequences,self.sequences_scores,self.scores,self.logits,self.beam_indices,self.encoder_attentions,self.encoder_hidden_states,self.decoder_attentions,self.cross_attentions,self.decoder_hidden_states,self.past_key_values,)==(other.sequences,other.sequences_scores,other.scores,other.logits,other.beam_indices,other.encoder_attentions,other.encoder_hidden_states,other.decoder_attentions,other.cross_attentions,other.decoder_hidden_states,other.past_key_values,)
  return NotImplemented
 return __eq__