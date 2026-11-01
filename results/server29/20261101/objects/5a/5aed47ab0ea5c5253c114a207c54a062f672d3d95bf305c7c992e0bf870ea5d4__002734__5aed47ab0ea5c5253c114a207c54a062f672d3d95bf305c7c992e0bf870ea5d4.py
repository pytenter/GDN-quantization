def __create_fn__(_type_graph, _type_signature, _type_module_call_graph, _type_metadata, _type_treespec_namedtuple_fields, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_metadata, _dflt_treespec_namedtuple_fields, _return_type):
 def __init__(self,graph:_type_graph,signature:_type_signature,module_call_graph:_type_module_call_graph,metadata:_type_metadata=_HAS_DEFAULT_FACTORY,treespec_namedtuple_fields:_type_treespec_namedtuple_fields=_HAS_DEFAULT_FACTORY)->_return_type:
  self.graph=graph
  self.signature=signature
  self.module_call_graph=module_call_graph
  self.metadata=_dflt_metadata() if metadata is _HAS_DEFAULT_FACTORY else metadata
  self.treespec_namedtuple_fields=_dflt_treespec_namedtuple_fields() if treespec_namedtuple_fields is _HAS_DEFAULT_FACTORY else treespec_namedtuple_fields
 return __init__