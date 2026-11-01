def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.conversion_sources,self.first_detection_run_guid,self.first_detection_time_utc,self.invocation_index,self.last_detection_run_guid,self.last_detection_time_utc,self.properties,)==(other.conversion_sources,other.first_detection_run_guid,other.first_detection_time_utc,other.invocation_index,other.last_detection_run_guid,other.last_detection_time_utc,other.properties,)
  return NotImplemented
 return __eq__