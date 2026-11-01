def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.vt_id,self.source,)==(other.vt_id,other.source,)
  return NotImplemented
 return __eq__