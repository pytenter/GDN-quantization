def __create_fn__(_type_rule, _type_level, _type_message, _type_locations, _type_stacks, _type_graphs, _type_thread_flow_locations, _type_additional_messages, _type_tags, _type_source_exception, _type_logger, _type__current_log_section_depth, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_message, _dflt_locations, _dflt_stacks, _dflt_graphs, _dflt_thread_flow_locations, _dflt_additional_messages, _dflt_tags, _dflt_source_exception, _dflt__current_log_section_depth, _return_type):
 def __init__(self,rule:_type_rule,level:_type_level,message:_type_message=_dflt_message,locations:_type_locations=_HAS_DEFAULT_FACTORY,stacks:_type_stacks=_HAS_DEFAULT_FACTORY,graphs:_type_graphs=_HAS_DEFAULT_FACTORY,thread_flow_locations:_type_thread_flow_locations=_HAS_DEFAULT_FACTORY,additional_messages:_type_additional_messages=_HAS_DEFAULT_FACTORY,tags:_type_tags=_HAS_DEFAULT_FACTORY,source_exception:_type_source_exception=_dflt_source_exception,_current_log_section_depth:_type__current_log_section_depth=_dflt__current_log_section_depth)->_return_type:
  self.rule=rule
  self.level=level
  self.message=message
  self.locations=_dflt_locations() if locations is _HAS_DEFAULT_FACTORY else locations
  self.stacks=_dflt_stacks() if stacks is _HAS_DEFAULT_FACTORY else stacks
  self.graphs=_dflt_graphs() if graphs is _HAS_DEFAULT_FACTORY else graphs
  self.thread_flow_locations=_dflt_thread_flow_locations() if thread_flow_locations is _HAS_DEFAULT_FACTORY else thread_flow_locations
  self.additional_messages=_dflt_additional_messages() if additional_messages is _HAS_DEFAULT_FACTORY else additional_messages
  self.tags=_dflt_tags() if tags is _HAS_DEFAULT_FACTORY else tags
  self.source_exception=source_exception
  self._current_log_section_depth=_current_log_section_depth
  self.__post_init__()
 return __init__