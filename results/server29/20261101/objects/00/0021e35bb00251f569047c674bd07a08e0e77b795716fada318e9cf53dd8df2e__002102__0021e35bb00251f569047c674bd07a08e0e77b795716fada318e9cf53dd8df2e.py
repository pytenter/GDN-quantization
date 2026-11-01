def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.rule,self.level,self.message,self.locations,self.stacks,self.graphs,self.thread_flow_locations,self.additional_messages,self.tags,self.source_exception,self.logger,self._current_log_section_depth,)==(other.rule,other.level,other.message,other.locations,other.stacks,other.graphs,other.thread_flow_locations,other.additional_messages,other.tags,other.source_exception,other.logger,other._current_log_section_depth,)
  return NotImplemented
 return __eq__