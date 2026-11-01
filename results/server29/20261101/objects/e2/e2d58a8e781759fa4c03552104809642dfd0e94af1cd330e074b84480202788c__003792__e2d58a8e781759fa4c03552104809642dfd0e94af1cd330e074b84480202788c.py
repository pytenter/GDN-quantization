def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.key,self.value,self.description,self.updated_at,)==(other.key,other.value,other.description,other.updated_at,)
  return NotImplemented
 return __eq__