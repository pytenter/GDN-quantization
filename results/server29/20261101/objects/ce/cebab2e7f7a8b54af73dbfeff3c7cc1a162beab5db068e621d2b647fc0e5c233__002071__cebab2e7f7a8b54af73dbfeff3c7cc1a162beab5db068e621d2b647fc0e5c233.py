def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.runs,self.version,self.schema_uri,self.inline_external_properties,self.properties,)==(other.runs,other.version,other.schema_uri,other.inline_external_properties,other.properties,)
  return NotImplemented
 return __eq__