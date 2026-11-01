def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.fw_compiler,self.bw_compiler,self.partition_fn,self.decompositions,self.num_params_buffers,self.aot_id,self.keep_inference_input_mutations,self.is_export,self.no_tangents,self.dynamic_shapes,self.aot_autograd_arg_pos_to_source,self.static_input_indices,self.inference_compiler,self.enable_log,self.pre_dispatch,self.cache_info,)==(other.fw_compiler,other.bw_compiler,other.partition_fn,other.decompositions,other.num_params_buffers,other.aot_id,other.keep_inference_input_mutations,other.is_export,other.no_tangents,other.dynamic_shapes,other.aot_autograd_arg_pos_to_source,other.static_input_indices,other.inference_compiler,other.enable_log,other.pre_dispatch,other.cache_info,)
  return NotImplemented
 return __eq__