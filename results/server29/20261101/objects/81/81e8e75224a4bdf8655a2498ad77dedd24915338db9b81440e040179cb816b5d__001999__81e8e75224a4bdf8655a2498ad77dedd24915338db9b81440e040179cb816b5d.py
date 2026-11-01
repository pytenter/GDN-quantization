def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.message,self.associated_rule,self.descriptor,self.exception,self.level,self.locations,self.properties,self.thread_id,self.time_utc,)==(other.message,other.associated_rule,other.descriptor,other.exception,other.level,other.locations,other.properties,other.thread_id,other.time_utc,)
  return NotImplemented
 return __eq__