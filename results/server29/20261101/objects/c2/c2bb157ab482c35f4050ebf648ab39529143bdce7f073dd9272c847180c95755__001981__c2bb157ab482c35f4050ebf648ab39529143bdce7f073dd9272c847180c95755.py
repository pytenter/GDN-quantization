def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.execution_order,self.execution_time_utc,self.importance,self.index,self.kinds,self.location,self.module,self.nesting_level,self.properties,self.stack,self.state,self.taxa,self.web_request,self.web_response,)==(other.execution_order,other.execution_time_utc,other.importance,other.index,other.kinds,other.location,other.module,other.nesting_level,other.properties,other.stack,other.state,other.taxa,other.web_request,other.web_response,)
  return NotImplemented
 return __eq__