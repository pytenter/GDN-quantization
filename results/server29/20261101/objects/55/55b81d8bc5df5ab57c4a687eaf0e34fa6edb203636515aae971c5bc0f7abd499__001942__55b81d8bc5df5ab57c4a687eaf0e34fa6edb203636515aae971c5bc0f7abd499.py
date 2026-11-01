def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.artifact_location,self.replacements,self.properties,)==(other.artifact_location,other.replacements,other.properties,)
  return NotImplemented
 return __eq__