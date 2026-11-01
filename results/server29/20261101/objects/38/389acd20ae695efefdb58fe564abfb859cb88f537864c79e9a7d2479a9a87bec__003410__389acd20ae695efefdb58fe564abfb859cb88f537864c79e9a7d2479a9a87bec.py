def __create_fn__(_type_nodes, _type_arg_ancestor_nodes, _type_A_node, _type_B_node, _type_A_scale_node, _type_B_scale_node, _type_bias_node, _type_result_scale_node, _type_out_dtype, _type_use_fast_accum, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _return_type):
 def __init__(self,nodes:_type_nodes,A_node:_type_A_node,B_node:_type_B_node,A_scale_node:_type_A_scale_node,B_scale_node:_type_B_scale_node,bias_node:_type_bias_node,result_scale_node:_type_result_scale_node,out_dtype:_type_out_dtype,use_fast_accum:_type_use_fast_accum)->_return_type:
  self.nodes=nodes
  self.A_node=A_node
  self.B_node=B_node
  self.A_scale_node=A_scale_node
  self.B_scale_node=B_scale_node
  self.bias_node=bias_node
  self.result_scale_node=result_scale_node
  self.out_dtype=out_dtype
  self.use_fast_accum=use_fast_accum
  self.__post_init__()
 return __init__