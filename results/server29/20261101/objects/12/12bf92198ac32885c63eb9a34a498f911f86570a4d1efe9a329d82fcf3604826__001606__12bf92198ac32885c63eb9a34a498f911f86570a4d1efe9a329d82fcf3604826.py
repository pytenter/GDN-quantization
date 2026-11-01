def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.module,self.accessed_attrs,)==(other.module,other.accessed_attrs,)
  return NotImplemented
 return __eq__