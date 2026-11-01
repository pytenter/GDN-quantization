def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.partitioning_done,self.old_num_fw_outputs,self.old_num_fw_inputs,self.new_fw_hop_gm,self.new_bw_hop_gm,self.new_num_sym_nodes,self.new_num_saved_nodes,)==(other.partitioning_done,other.old_num_fw_outputs,other.old_num_fw_inputs,other.new_fw_hop_gm,other.new_bw_hop_gm,other.new_num_sym_nodes,other.new_num_saved_nodes,)
  return NotImplemented
 return __eq__