def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.configuration,self.descriptor,self.properties,)==(other.configuration,other.descriptor,other.properties,)
  return NotImplemented
 return __eq__