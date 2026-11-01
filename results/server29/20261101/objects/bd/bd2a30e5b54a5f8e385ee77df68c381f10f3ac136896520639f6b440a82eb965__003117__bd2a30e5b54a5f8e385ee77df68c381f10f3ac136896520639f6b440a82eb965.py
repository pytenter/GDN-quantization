def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.node_name_to_qconfig,self.node_name_to_scope,self.prepare_custom_config,self.equalization_node_name_to_qconfig,self.qconfig_mapping,self.is_qat,self.observed_node_names,self.is_observed_standalone_module,self.standalone_module_input_quantized_idxs,self.standalone_module_output_quantized_idxs,)==(other.node_name_to_qconfig,other.node_name_to_scope,other.prepare_custom_config,other.equalization_node_name_to_qconfig,other.qconfig_mapping,other.is_qat,other.observed_node_names,other.is_observed_standalone_module,other.standalone_module_input_quantized_idxs,other.standalone_module_output_quantized_idxs,)
  return NotImplemented
 return __eq__