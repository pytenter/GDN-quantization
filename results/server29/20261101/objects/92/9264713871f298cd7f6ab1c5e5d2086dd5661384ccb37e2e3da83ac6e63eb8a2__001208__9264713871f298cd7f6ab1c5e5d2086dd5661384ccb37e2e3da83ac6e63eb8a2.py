def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.warn_only,self.vr,)==(other.warn_only,other.vr,)
  return NotImplemented
 return __eq__