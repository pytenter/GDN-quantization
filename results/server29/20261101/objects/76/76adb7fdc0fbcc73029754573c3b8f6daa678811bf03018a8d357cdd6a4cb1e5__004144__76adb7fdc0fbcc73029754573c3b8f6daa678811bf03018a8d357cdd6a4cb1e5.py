def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.loss,self.logits,self.encoder_z_loss,self.decoder_z_loss,self.encoder_aux_loss,self.decoder_aux_loss,self.past_key_values,self.decoder_hidden_states,self.decoder_attentions,self.decoder_router_logits,self.cross_attentions,self.encoder_last_hidden_state,self.encoder_hidden_states,self.encoder_attentions,self.encoder_router_logits,)==(other.loss,other.logits,other.encoder_z_loss,other.decoder_z_loss,other.encoder_aux_loss,other.decoder_aux_loss,other.past_key_values,other.decoder_hidden_states,other.decoder_attentions,other.decoder_router_logits,other.cross_attentions,other.encoder_last_hidden_state,other.encoder_hidden_states,other.encoder_attentions,other.encoder_router_logits,)
  return NotImplemented
 return __eq__