def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.guid,self.item_count,self.location,self.properties,)==(other.guid,other.item_count,other.location,other.properties,)
  return NotImplemented
 return __eq__