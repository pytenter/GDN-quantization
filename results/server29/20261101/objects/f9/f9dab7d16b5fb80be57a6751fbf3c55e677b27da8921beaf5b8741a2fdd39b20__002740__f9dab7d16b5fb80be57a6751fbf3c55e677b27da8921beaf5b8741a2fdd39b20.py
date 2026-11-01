def __create_fn__(_type_graph_module, _type_opset_version, _type_range_constraints, _type_schema_version, _type_verifiers, _type_torch_version, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_verifiers, _dflt_torch_version, _return_type):
 def __init__(self,graph_module:_type_graph_module,opset_version:_type_opset_version,range_constraints:_type_range_constraints,schema_version:_type_schema_version,verifiers:_type_verifiers=_HAS_DEFAULT_FACTORY,torch_version:_type_torch_version=_dflt_torch_version)->_return_type:
  self.graph_module=graph_module
  self.opset_version=opset_version
  self.range_constraints=range_constraints
  self.schema_version=schema_version
  self.verifiers=_dflt_verifiers() if verifiers is _HAS_DEFAULT_FACTORY else verifiers
  self.torch_version=torch_version
 return __init__