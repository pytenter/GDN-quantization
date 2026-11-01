def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.body,self.headers,self.index,self.method,self.parameters,self.properties,self.protocol,self.target,self.version,)==(other.body,other.headers,other.index,other.method,other.parameters,other.properties,other.protocol,other.target,other.version,)
  return NotImplemented
 return __eq__