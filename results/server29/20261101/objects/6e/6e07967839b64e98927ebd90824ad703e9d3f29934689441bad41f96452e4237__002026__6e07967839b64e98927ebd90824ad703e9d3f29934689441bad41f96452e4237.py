def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.edge_id,self.final_state,self.message,self.properties,self.step_over_edge_count,)==(other.edge_id,other.final_state,other.message,other.properties,other.step_over_edge_count,)
  return NotImplemented
 return __eq__