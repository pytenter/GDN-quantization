def __create_fn__(_type_backend, _type_mode, _type_fullgraph, _type_dynamic, _type_options, _type_disable, _type_use_regional_compilation, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_backend, _dflt_mode, _dflt_fullgraph, _dflt_dynamic, _dflt_options, _dflt_disable, _dflt_use_regional_compilation, _return_type):
 def __init__(self,backend:_type_backend=_dflt_backend,mode:_type_mode=_dflt_mode,fullgraph:_type_fullgraph=_dflt_fullgraph,dynamic:_type_dynamic=_dflt_dynamic,options:_type_options=_dflt_options,disable:_type_disable=_dflt_disable,use_regional_compilation:_type_use_regional_compilation=_dflt_use_regional_compilation)->_return_type:
  self.backend=backend
  self.mode=mode
  self.fullgraph=fullgraph
  self.dynamic=dynamic
  self.options=options
  self.disable=disable
  self.use_regional_compilation=use_regional_compilation
  self.__post_init__()
 return __init__