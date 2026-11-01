def __create_fn__(_type_anchors, _type_nodes_map, _type_placeholder_nodes, _type_returning_nodes, _type_name_node_map, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_nodes_map, _dflt_placeholder_nodes, _dflt_returning_nodes, _dflt_name_node_map, _return_type):
 def __init__(self,anchors:_type_anchors,nodes_map:_type_nodes_map=_HAS_DEFAULT_FACTORY,placeholder_nodes:_type_placeholder_nodes=_HAS_DEFAULT_FACTORY,returning_nodes:_type_returning_nodes=_HAS_DEFAULT_FACTORY,name_node_map:_type_name_node_map=_HAS_DEFAULT_FACTORY)->_return_type:
  self.anchors=anchors
  self.nodes_map=_dflt_nodes_map() if nodes_map is _HAS_DEFAULT_FACTORY else nodes_map
  self.placeholder_nodes=_dflt_placeholder_nodes() if placeholder_nodes is _HAS_DEFAULT_FACTORY else placeholder_nodes
  self.returning_nodes=_dflt_returning_nodes() if returning_nodes is _HAS_DEFAULT_FACTORY else returning_nodes
  self.name_node_map=_dflt_name_node_map() if name_node_map is _HAS_DEFAULT_FACTORY else name_node_map
 return __init__