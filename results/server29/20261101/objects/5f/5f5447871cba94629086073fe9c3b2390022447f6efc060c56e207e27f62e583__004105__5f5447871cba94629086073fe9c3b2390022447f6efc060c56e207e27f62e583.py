def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.loss,self.logits,self.past_key_values,self.hidden_states,self.attentions,self.z_loss,self.aux_loss,self.router_logits,)==(other.loss,other.logits,other.past_key_values,other.hidden_states,other.attentions,other.z_loss,other.aux_loss,other.router_logits,)
  return NotImplemented
 return __eq__