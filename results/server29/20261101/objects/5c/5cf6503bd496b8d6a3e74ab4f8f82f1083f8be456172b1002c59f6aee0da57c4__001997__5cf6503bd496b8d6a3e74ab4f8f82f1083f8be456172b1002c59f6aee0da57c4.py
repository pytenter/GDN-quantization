def __create_fn__(_type_message, _type_associated_rule, _type_descriptor, _type_exception, _type_level, _type_locations, _type_properties, _type_thread_id, _type_time_utc, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_associated_rule, _dflt_descriptor, _dflt_exception, _dflt_level, _dflt_locations, _dflt_properties, _dflt_thread_id, _dflt_time_utc, _return_type):
 def __init__(self,message:_type_message,associated_rule:_type_associated_rule=_dflt_associated_rule,descriptor:_type_descriptor=_dflt_descriptor,exception:_type_exception=_dflt_exception,level:_type_level=_dflt_level,locations:_type_locations=_dflt_locations,properties:_type_properties=_dflt_properties,thread_id:_type_thread_id=_dflt_thread_id,time_utc:_type_time_utc=_dflt_time_utc)->_return_type:
  self.message=message
  self.associated_rule=associated_rule
  self.descriptor=descriptor
  self.exception=exception
  self.level=level
  self.locations=locations
  self.properties=properties
  self.thread_id=thread_id
  self.time_utc=time_utc
 return __init__