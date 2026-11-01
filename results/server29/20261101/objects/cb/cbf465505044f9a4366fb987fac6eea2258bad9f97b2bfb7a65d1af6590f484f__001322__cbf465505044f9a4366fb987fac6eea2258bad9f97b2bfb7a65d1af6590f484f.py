def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.code,self.instructions,self.prefix_block_target_offset_remap,self.block_target_offset_remap,)==(other.code,other.instructions,other.prefix_block_target_offset_remap,other.block_target_offset_remap,)
  return NotImplemented
 return __eq__