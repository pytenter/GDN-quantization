def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.description,self.index,self.properties,self.uri,self.uri_base_id,)==(other.description,other.index,other.properties,other.uri,other.uri_base_id,)
  return NotImplemented
 return __eq__