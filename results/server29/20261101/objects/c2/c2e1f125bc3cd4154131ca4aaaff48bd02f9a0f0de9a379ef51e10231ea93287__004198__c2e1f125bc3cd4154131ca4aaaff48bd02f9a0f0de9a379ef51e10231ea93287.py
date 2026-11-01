def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.last_hidden_state,self.past_key_values,self.decoder_hidden_states,self.decoder_attentions,self.cross_attentions,self.encoder_last_hidden_state,self.encoder_hidden_states,self.encoder_attentions,self.loc,self.scale,self.static_features,)==(other.last_hidden_state,other.past_key_values,other.decoder_hidden_states,other.decoder_attentions,other.cross_attentions,other.encoder_last_hidden_state,other.encoder_hidden_states,other.encoder_attentions,other.loc,other.scale,other.static_features,)
  return NotImplemented
 return __eq__