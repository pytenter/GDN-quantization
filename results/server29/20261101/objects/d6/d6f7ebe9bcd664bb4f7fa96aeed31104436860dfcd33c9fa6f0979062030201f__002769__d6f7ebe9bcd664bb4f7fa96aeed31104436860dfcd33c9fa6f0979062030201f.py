def __create_fn__(_type_graph_module, _type_signature, _type_module_call_graph, _type_names_to_symbols, _type_state_dict, _type_constants, _type_example_inputs, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _return_type):
 def __init__(self,graph_module:_type_graph_module,signature:_type_signature,module_call_graph:_type_module_call_graph,names_to_symbols:_type_names_to_symbols,state_dict:_type_state_dict,constants:_type_constants,example_inputs:_type_example_inputs)->_return_type:
  self.graph_module=graph_module
  self.signature=signature
  self.module_call_graph=module_call_graph
  self.names_to_symbols=names_to_symbols
  self.state_dict=state_dict
  self.constants=constants
  self.example_inputs=example_inputs
 return __init__