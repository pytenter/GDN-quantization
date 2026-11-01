def __create_fn__(_type_reason, _type_user_stack, _type_graph_break, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_graph_break, _return_type):
 def __init__(self,reason:_type_reason,user_stack:_type_user_stack,graph_break:_type_graph_break=_dflt_graph_break)->_return_type:
  self.reason=reason
  self.user_stack=user_stack
  self.graph_break=graph_break
  self.__post_init__()
 return __init__