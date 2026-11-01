def __create_fn__(_type_name, _type_graph_module, _type_graph, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_graph, _return_type):
 def __init__(self,*,name:_type_name,graph_module:_type_graph_module,graph:_type_graph=_dflt_graph)->_return_type:
  self.name=name
  self.graph_module=graph_module
  self.graph=graph
  self.__post_init__()
 return __init__