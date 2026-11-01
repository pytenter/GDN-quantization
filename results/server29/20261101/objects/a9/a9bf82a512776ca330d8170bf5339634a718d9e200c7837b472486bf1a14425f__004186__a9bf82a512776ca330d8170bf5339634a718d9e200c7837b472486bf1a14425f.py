def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.loss,self.logits,self.embeddings,self.hidden_states,self.attentions,)==(other.loss,other.logits,other.embeddings,other.hidden_states,other.attentions,)
  return NotImplemented
 return __eq__