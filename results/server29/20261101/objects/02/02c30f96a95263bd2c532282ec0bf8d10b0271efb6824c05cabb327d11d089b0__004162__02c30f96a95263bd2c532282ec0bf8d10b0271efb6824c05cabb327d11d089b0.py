def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.loss,self.start_logits,self.end_logits,self.hidden_states,self.attentions,)==(other.loss,other.start_logits,other.end_logits,other.hidden_states,other.attentions,)
  return NotImplemented
 return __eq__