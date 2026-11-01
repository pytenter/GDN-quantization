def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.output,self.final_state,self.stage_tensors,)==(other.output,other.final_state,other.stage_tensors,)
  return NotImplemented
 return __eq__