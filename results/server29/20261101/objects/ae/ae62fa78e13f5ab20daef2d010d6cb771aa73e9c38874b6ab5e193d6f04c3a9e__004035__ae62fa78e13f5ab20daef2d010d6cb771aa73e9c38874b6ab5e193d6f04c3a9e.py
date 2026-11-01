def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.sp_seq_length,self.sp_seq_length_is_variable,self.sp_attn_implementation,)==(other.sp_seq_length,other.sp_seq_length_is_variable,other.sp_attn_implementation,)
  return NotImplemented
 return __eq__