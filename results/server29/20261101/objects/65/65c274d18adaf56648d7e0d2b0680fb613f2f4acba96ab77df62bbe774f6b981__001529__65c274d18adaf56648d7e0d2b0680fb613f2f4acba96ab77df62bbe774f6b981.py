def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.source_name,)==(other.source_name,)
  return NotImplemented
 return __eq__