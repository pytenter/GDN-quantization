def __create_fn__(_type_size_alloc, _type_size_free, _type_succ_nodes, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_size_alloc, _dflt_size_free, _dflt_succ_nodes, _return_type):
 def __init__(self,size_alloc:_type_size_alloc=_dflt_size_alloc,size_free:_type_size_free=_dflt_size_free,succ_nodes:_type_succ_nodes=_HAS_DEFAULT_FACTORY)->_return_type:
  self.size_alloc=size_alloc
  self.size_free=size_free
  self.succ_nodes=_dflt_succ_nodes() if succ_nodes is _HAS_DEFAULT_FACTORY else succ_nodes
 return __init__