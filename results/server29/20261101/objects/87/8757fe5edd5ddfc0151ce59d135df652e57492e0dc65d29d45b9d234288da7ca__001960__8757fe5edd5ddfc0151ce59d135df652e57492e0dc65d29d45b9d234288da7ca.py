def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.annotations,self.id,self.logical_locations,self.message,self.physical_location,self.properties,self.relationships,)==(other.annotations,other.id,other.logical_locations,other.message,other.physical_location,other.properties,other.relationships,)
  return NotImplemented
 return __eq__