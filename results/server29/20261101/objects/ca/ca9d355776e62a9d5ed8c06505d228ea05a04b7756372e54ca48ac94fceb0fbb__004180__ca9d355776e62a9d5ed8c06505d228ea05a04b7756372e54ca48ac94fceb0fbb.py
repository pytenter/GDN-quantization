def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.loss,self.reconstruction,self.hidden_states,self.attentions,)==(other.loss,other.reconstruction,other.hidden_states,other.attentions,)
  return NotImplemented
 return __eq__