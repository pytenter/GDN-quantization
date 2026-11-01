def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.compiled_fw,self.compiled_bw,self.aot_joint_graph_str,self.aot_forward_graph_str,self.aot_backward_graph_str,self.runtime_metadata,self.dispatch_wrappers,self.maybe_subclass_meta,self.num_fw_outs_saved_for_bw,self.indices_of_inps_to_detach,self.forward_time_taken_ns,self.backward_time_taken_ns,)==(other.compiled_fw,other.compiled_bw,other.aot_joint_graph_str,other.aot_forward_graph_str,other.aot_backward_graph_str,other.runtime_metadata,other.dispatch_wrappers,other.maybe_subclass_meta,other.num_fw_outs_saved_for_bw,other.indices_of_inps_to_detach,other.forward_time_taken_ns,other.backward_time_taken_ns,)
  return NotImplemented
 return __eq__