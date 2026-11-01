def __create_fn__(_type_inputs, _type_outputs, _type_nodes, _type_tensor_values, _type_sym_int_values, _type_sym_bool_values, _type_sym_float_values, _type_is_single_tensor_return, _type_custom_obj_values, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_inputs, _dflt_outputs, _dflt_nodes, _dflt_tensor_values, _dflt_sym_int_values, _dflt_sym_bool_values, _dflt_sym_float_values, _dflt_is_single_tensor_return, _dflt_custom_obj_values, _return_type):
 def __init__(self,inputs:_type_inputs=_HAS_DEFAULT_FACTORY,outputs:_type_outputs=_HAS_DEFAULT_FACTORY,nodes:_type_nodes=_HAS_DEFAULT_FACTORY,tensor_values:_type_tensor_values=_HAS_DEFAULT_FACTORY,sym_int_values:_type_sym_int_values=_HAS_DEFAULT_FACTORY,sym_bool_values:_type_sym_bool_values=_HAS_DEFAULT_FACTORY,sym_float_values:_type_sym_float_values=_HAS_DEFAULT_FACTORY,is_single_tensor_return:_type_is_single_tensor_return=_dflt_is_single_tensor_return,custom_obj_values:_type_custom_obj_values=_HAS_DEFAULT_FACTORY)->_return_type:
  self.inputs=_dflt_inputs() if inputs is _HAS_DEFAULT_FACTORY else inputs
  self.outputs=_dflt_outputs() if outputs is _HAS_DEFAULT_FACTORY else outputs
  self.nodes=_dflt_nodes() if nodes is _HAS_DEFAULT_FACTORY else nodes
  self.tensor_values=_dflt_tensor_values() if tensor_values is _HAS_DEFAULT_FACTORY else tensor_values
  self.sym_int_values=_dflt_sym_int_values() if sym_int_values is _HAS_DEFAULT_FACTORY else sym_int_values
  self.sym_bool_values=_dflt_sym_bool_values() if sym_bool_values is _HAS_DEFAULT_FACTORY else sym_bool_values
  self.sym_float_values=_dflt_sym_float_values() if sym_float_values is _HAS_DEFAULT_FACTORY else sym_float_values
  self.is_single_tensor_return=is_single_tensor_return
  self.custom_obj_values=_dflt_custom_obj_values() if custom_obj_values is _HAS_DEFAULT_FACTORY else custom_obj_values
 return __init__