def __create_fn__(_type_wrapper, _type_graph, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _return_type):
 def __init__(self,wrapper:_type_wrapper,graph:_type_graph)->_return_type:
  self.wrapper=wrapper
  self.graph=graph
  self.__post_init__()
 return __init__