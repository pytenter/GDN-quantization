def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.base,self.idx_key,self.is_kw,)==(other.base,other.idx_key,other.is_kw,)
  return NotImplemented
 return __eq__