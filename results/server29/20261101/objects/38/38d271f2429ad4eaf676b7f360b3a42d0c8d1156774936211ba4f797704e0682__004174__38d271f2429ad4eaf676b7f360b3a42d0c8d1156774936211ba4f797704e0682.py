def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.loss,self.logits,self.hidden_states,)==(other.loss,other.logits,other.hidden_states,)
  return NotImplemented
 return __eq__