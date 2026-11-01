def __create_fn__(_type_shape, _type_node_list, _type_inputs, _type_wait_nodes, _type_comm_node, _type_outputs, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _return_type):
 def __init__(self,shape:_type_shape,node_list:_type_node_list,inputs:_type_inputs,wait_nodes:_type_wait_nodes,comm_node:_type_comm_node,outputs:_type_outputs)->_return_type:
  self.shape=shape
  self.node_list=node_list
  self.inputs=inputs
  self.wait_nodes=wait_nodes
  self.comm_node=comm_node
  self.outputs=outputs
 return __init__