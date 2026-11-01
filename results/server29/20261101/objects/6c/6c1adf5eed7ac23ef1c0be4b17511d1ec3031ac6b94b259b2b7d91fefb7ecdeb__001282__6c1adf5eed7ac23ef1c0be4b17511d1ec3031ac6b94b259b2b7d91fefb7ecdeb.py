def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.loc,self.locals,self.symbols,)==(other.loc,other.locals,other.symbols,)
  return NotImplemented
 return __eq__