def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.artifact_changes,self.description,self.properties,)==(other.artifact_changes,other.description,other.properties,)
  return NotImplemented
 return __eq__