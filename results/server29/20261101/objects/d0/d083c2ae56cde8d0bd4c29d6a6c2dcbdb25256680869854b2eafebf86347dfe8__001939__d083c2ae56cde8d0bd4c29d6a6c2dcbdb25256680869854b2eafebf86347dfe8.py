def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.deleted_region,self.inserted_content,self.properties,)==(other.deleted_region,other.inserted_content,other.properties,)
  return NotImplemented
 return __eq__