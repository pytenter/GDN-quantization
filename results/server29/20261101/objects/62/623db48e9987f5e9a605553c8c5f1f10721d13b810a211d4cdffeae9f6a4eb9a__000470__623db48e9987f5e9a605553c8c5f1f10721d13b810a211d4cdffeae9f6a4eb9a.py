def __create_fn__(_type_log_qname_to_level, _type_artifact_names, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_log_qname_to_level, _dflt_artifact_names, _return_type):
 def __init__(self,log_qname_to_level:_type_log_qname_to_level=_HAS_DEFAULT_FACTORY,artifact_names:_type_artifact_names=_HAS_DEFAULT_FACTORY)->_return_type:
  self.log_qname_to_level=_dflt_log_qname_to_level() if log_qname_to_level is _HAS_DEFAULT_FACTORY else log_qname_to_level
  self.artifact_names=_dflt_artifact_names() if artifact_names is _HAS_DEFAULT_FACTORY else artifact_names
 return __init__