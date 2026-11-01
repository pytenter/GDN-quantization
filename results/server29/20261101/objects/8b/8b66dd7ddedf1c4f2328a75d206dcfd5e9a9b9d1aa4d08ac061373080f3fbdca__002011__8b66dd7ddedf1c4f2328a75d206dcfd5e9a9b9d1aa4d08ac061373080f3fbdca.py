def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.name,self.download_uri,self.full_description,self.full_name,self.information_uri,self.properties,self.short_description,)==(other.name,other.download_uri,other.full_description,other.full_name,other.information_uri,other.properties,other.short_description,)
  return NotImplemented
 return __eq__