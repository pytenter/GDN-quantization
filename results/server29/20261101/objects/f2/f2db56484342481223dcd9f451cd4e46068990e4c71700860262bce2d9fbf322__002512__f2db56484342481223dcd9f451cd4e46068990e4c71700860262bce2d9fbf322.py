def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.fx_graph_cache_key,self.backward_state_indices,self.num_symints_saved_for_bw_,)==(other.fx_graph_cache_key,other.backward_state_indices,other.num_symints_saved_for_bw_,)
  return NotImplemented
 return __eq__