def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.type,self.key,self.content,)==(other.type,other.key,other.content,)
  return NotImplemented
 return __eq__