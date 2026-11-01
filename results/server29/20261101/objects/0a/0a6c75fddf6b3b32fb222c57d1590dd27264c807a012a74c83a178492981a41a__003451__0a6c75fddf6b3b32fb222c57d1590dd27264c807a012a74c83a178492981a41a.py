def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.ht,self.offsets,self.q_prev,self.k_prev,self.v_prev,self.g_prev,self.level_scales_prev,)==(other.ht,other.offsets,other.q_prev,other.k_prev,other.v_prev,other.g_prev,other.level_scales_prev,)
  return NotImplemented
 return __eq__