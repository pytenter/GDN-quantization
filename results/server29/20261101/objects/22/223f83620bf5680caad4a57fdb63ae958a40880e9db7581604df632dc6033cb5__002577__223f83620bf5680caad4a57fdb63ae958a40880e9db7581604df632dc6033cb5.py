def __create_fn__(_type_inputs, _type__required_fw_nodes, _type_required_bw_nodes, _type_unclaimed_nodes, _type_fw_order, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _return_type):
 def __init__(self,inputs:_type_inputs,_required_fw_nodes:_type__required_fw_nodes,required_bw_nodes:_type_required_bw_nodes,unclaimed_nodes:_type_unclaimed_nodes,fw_order:_type_fw_order)->_return_type:
  self.inputs=inputs
  self._required_fw_nodes=_required_fw_nodes
  self.required_bw_nodes=required_bw_nodes
  self.unclaimed_nodes=unclaimed_nodes
  self.fw_order=fw_order
 return __init__