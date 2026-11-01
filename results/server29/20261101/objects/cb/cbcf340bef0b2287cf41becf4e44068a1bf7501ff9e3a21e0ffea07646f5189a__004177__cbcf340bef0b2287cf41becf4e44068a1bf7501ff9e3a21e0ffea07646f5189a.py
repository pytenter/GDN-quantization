def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.loss,self.predicted_depth,self.hidden_states,self.attentions,)==(other.loss,other.predicted_depth,other.hidden_states,other.attentions,)
  return NotImplemented
 return __eq__