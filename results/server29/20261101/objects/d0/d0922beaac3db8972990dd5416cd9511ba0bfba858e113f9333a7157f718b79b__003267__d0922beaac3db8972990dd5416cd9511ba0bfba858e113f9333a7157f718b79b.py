def __create_fn__(_type_body, _type_template_mask, _type_template_out, _type_compute, _type_indexing_code, _type_loads, _type_stores, _type_ops_handler, _type_range_trees, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_template_mask, _dflt_template_out, _dflt_compute, _dflt_indexing_code, _dflt_loads, _dflt_stores, _dflt_ops_handler, _dflt_range_trees, _return_type):
 def __init__(self,body:_type_body,template_mask:_type_template_mask=_dflt_template_mask,template_out:_type_template_out=_dflt_template_out,compute:_type_compute=_HAS_DEFAULT_FACTORY,indexing_code:_type_indexing_code=_HAS_DEFAULT_FACTORY,loads:_type_loads=_HAS_DEFAULT_FACTORY,stores:_type_stores=_HAS_DEFAULT_FACTORY,ops_handler:_type_ops_handler=_dflt_ops_handler,range_trees:_type_range_trees=_dflt_range_trees)->_return_type:
  self.body=body
  self.template_mask=template_mask
  self.template_out=template_out
  self.compute=_dflt_compute() if compute is _HAS_DEFAULT_FACTORY else compute
  self.indexing_code=_dflt_indexing_code() if indexing_code is _HAS_DEFAULT_FACTORY else indexing_code
  self.loads=_dflt_loads() if loads is _HAS_DEFAULT_FACTORY else loads
  self.stores=_dflt_stores() if stores is _HAS_DEFAULT_FACTORY else stores
  self.ops_handler=ops_handler
  self.range_trees=range_trees
  self.__post_init__()
 return __init__