def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.artifact_location,self.description,self.properties,self.rectangles,self.regions,)==(other.artifact_location,other.description,other.properties,other.rectangles,other.regions,)
  return NotImplemented
 return __eq__