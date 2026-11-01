def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.loss,self.aux_loss,self.logits,self.past_key_values,self.hidden_states,self.attentions,self.router_logits,)==(other.loss,other.aux_loss,other.logits,other.past_key_values,other.hidden_states,other.attentions,other.router_logits,)
  return NotImplemented
 return __eq__