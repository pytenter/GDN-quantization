def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.body,self.headers,self.index,self.no_response_received,self.properties,self.protocol,self.reason_phrase,self.status_code,self.version,)==(other.body,other.headers,other.index,other.no_response_received,other.properties,other.protocol,other.reason_phrase,other.status_code,other.version,)
  return NotImplemented
 return __eq__