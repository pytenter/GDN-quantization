def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.feature_maps,self.hidden_states,self.attentions,)==(other.feature_maps,other.hidden_states,other.attentions,)
  return NotImplemented
 return __eq__