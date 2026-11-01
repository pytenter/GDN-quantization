def __create_fn__(_type_level, _type_mode, _type_write_thread_id, _type_marker_name, _type_is_reader, _type_token, _type_heartbeat_thread, _type_heartbeat_stop, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _return_type):
 def __init__(self,level:_type_level,mode:_type_mode,write_thread_id:_type_write_thread_id,marker_name:_type_marker_name,is_reader:_type_is_reader,token:_type_token,heartbeat_thread:_type_heartbeat_thread,heartbeat_stop:_type_heartbeat_stop)->_return_type:
  self.level=level
  self.mode=mode
  self.write_thread_id=write_thread_id
  self.marker_name=marker_name
  self.is_reader=is_reader
  self.token=token
  self.heartbeat_thread=heartbeat_thread
  self.heartbeat_stop=heartbeat_stop
 return __init__