def __create_fn__(_type_graph, _type_block, _type_opset, _type_original_node, _type_params_dict, _type_env, _type_values_in_env, _type_new_nodes, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_new_nodes, _return_type):
 def __init__(self,graph:_type_graph,block:_type_block,opset:_type_opset,original_node:_type_original_node,params_dict:_type_params_dict,env:_type_env,values_in_env:_type_values_in_env,new_nodes:_type_new_nodes=_HAS_DEFAULT_FACTORY)->_return_type:
  self.graph=graph
  self.block=block
  self.opset=opset
  self.original_node=original_node
  self.params_dict=params_dict
  self.env=env
  self.values_in_env=values_in_env
  self.new_nodes=_dflt_new_nodes() if new_nodes is _HAS_DEFAULT_FACTORY else new_nodes
 return __init__