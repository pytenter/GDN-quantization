def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.trace_joint,self.fw_only,self.maybe_subclass_meta,self.num_fw_outs_saved_for_bw,)==(other.trace_joint,other.fw_only,other.maybe_subclass_meta,other.num_fw_outs_saved_for_bw,)
  return NotImplemented
 return __eq__