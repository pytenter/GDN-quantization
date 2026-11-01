def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.model,self.static_input_idxs,self.id,self.constants,self.placeholders,self.mutated_input_idxs,)==(other.model,other.static_input_idxs,other.id,other.constants,other.placeholders,other.mutated_input_idxs,)
  return NotImplemented
 return __eq__