def __create_fn__(_type_duration_time_ns, _type_self_time_ns, _type_idle_time_ns, _type_queue_depth, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_duration_time_ns, _dflt_self_time_ns, _dflt_idle_time_ns, _dflt_queue_depth, _return_type):
 def __init__(self,duration_time_ns:_type_duration_time_ns=_dflt_duration_time_ns,self_time_ns:_type_self_time_ns=_dflt_self_time_ns,idle_time_ns:_type_idle_time_ns=_dflt_idle_time_ns,queue_depth:_type_queue_depth=_dflt_queue_depth)->_return_type:
  self.duration_time_ns=duration_time_ns
  self.self_time_ns=self_time_ns
  self.idle_time_ns=idle_time_ns
  self.queue_depth=queue_depth
 return __init__