def __create_fn__(_type_name, _type_version, _type_options, _type_diagnostics, _type__inflight_diagnostics, _type__previous_log_level, _type_logger, _type__bound_diagnostic_type, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_options, _dflt_diagnostics, _dflt__inflight_diagnostics, _return_type):
 def __init__(self,name:_type_name,version:_type_version,options:_type_options=_HAS_DEFAULT_FACTORY)->_return_type:
  self.name=name
  self.version=version
  self.options=_dflt_options() if options is _HAS_DEFAULT_FACTORY else options
  self.diagnostics=_dflt_diagnostics()
  self._inflight_diagnostics=_dflt__inflight_diagnostics()
 return __init__