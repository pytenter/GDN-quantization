def __create_fn__(_type_nodes, _type_source, _type_input_nodes, _type_output_nodes, _type_params, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_input_nodes, _dflt_output_nodes, _dflt_params, _return_type):
 def __init__(self,nodes:_type_nodes,source:_type_source,input_nodes:_type_input_nodes=_HAS_DEFAULT_FACTORY,output_nodes:_type_output_nodes=_HAS_DEFAULT_FACTORY,params:_type_params=_HAS_DEFAULT_FACTORY)->_return_type:
  self.nodes=nodes
  self.source=source
  self.input_nodes=_dflt_input_nodes() if input_nodes is _HAS_DEFAULT_FACTORY else input_nodes
  self.output_nodes=_dflt_output_nodes() if output_nodes is _HAS_DEFAULT_FACTORY else output_nodes
  self.params=_dflt_params() if params is _HAS_DEFAULT_FACTORY else params
 return __init__