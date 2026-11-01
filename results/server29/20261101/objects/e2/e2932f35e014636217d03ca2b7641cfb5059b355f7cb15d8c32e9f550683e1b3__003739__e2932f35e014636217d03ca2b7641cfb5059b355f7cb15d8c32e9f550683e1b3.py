def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.file_hash,self.refresh_route,)==(other.file_hash,other.refresh_route,)
  return NotImplemented
 return __eq__