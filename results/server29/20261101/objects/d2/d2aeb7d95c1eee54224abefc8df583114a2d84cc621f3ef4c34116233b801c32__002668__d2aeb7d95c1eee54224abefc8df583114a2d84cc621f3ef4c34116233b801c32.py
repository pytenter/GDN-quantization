def __create_fn__(_type_inputs, _type_outputs, _type_nodes, _type_tensor_values, _type_sym_int_values, _type_sym_bool_values, _type_is_single_tensor_return, _type_custom_obj_values, _type_sym_float_values, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_is_single_tensor_return, _dflt_custom_obj_values, _dflt_sym_float_values, _return_type):
 def __init__(self,inputs:_type_inputs,outputs:_type_outputs,nodes:_type_nodes,tensor_values:_type_tensor_values,sym_int_values:_type_sym_int_values,sym_bool_values:_type_sym_bool_values,is_single_tensor_return:_type_is_single_tensor_return=_dflt_is_single_tensor_return,custom_obj_values:_type_custom_obj_values=_HAS_DEFAULT_FACTORY,sym_float_values:_type_sym_float_values=_HAS_DEFAULT_FACTORY)->_return_type:
  self.inputs=inputs
  self.outputs=outputs
  self.nodes=nodes
  self.tensor_values=tensor_values
  self.sym_int_values=sym_int_values
  self.sym_bool_values=sym_bool_values
  self.is_single_tensor_return=is_single_tensor_return
  self.custom_obj_values=_dflt_custom_obj_values() if custom_obj_values is _HAS_DEFAULT_FACTORY else custom_obj_values
  self.sym_float_values=_dflt_sym_float_values() if sym_float_values is _HAS_DEFAULT_FACTORY else sym_float_values
 return __init__