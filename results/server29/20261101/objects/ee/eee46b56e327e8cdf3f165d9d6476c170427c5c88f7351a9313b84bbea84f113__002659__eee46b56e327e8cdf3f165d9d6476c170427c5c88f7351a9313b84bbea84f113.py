def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.name,self.class_fqn,)==(other.name,other.class_fqn,)
  return NotImplemented
 return __eq__