def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.address,self.artifact_location,self.context_region,self.properties,self.region,)==(other.address,other.artifact_location,other.context_region,other.properties,other.region,)
  return NotImplemented
 return __eq__