def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.ban_if_used_far_apart,self.ban_if_long_fusible_chains,self.ban_if_materialized_backward,self.ban_if_not_in_allowlist,self.ban_if_reduction,)==(other.ban_if_used_far_apart,other.ban_if_long_fusible_chains,other.ban_if_materialized_backward,other.ban_if_not_in_allowlist,other.ban_if_reduction,)
  return NotImplemented
 return __eq__