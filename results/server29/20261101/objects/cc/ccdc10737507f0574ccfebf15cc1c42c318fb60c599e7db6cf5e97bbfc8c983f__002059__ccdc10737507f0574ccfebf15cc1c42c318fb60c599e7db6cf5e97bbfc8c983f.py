def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.correlation_guid,self.description,self.guid,self.id,self.properties,)==(other.correlation_guid,other.description,other.guid,other.id,other.properties,)
  return NotImplemented
 return __eq__