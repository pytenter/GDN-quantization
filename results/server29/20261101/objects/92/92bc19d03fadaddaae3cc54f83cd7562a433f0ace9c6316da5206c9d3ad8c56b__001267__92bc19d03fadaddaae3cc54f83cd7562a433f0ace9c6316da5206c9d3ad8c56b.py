def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.exprs,self.source_to_symbol,)==(other.exprs,other.source_to_symbol,)
  return NotImplemented
 return __eq__