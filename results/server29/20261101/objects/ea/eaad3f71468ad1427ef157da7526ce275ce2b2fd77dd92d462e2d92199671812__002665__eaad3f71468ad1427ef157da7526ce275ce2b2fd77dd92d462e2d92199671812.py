def __create_fn__(_type_target, _type_inputs, _type_outputs, _type_metadata, _type_is_hop_single_tensor_return, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_is_hop_single_tensor_return, _return_type):
 def __init__(self,target:_type_target,inputs:_type_inputs,outputs:_type_outputs,metadata:_type_metadata,is_hop_single_tensor_return:_type_is_hop_single_tensor_return=_dflt_is_hop_single_tensor_return)->_return_type:
  self.target=target
  self.inputs=inputs
  self.outputs=outputs
  self.metadata=metadata
  self.is_hop_single_tensor_return=is_hop_single_tensor_return
 return __init__