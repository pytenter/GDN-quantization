def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.argnames,self.initialnames,self.names_closure,self.name2fixturedefs,)==(other.argnames,other.initialnames,other.names_closure,other.name2fixturedefs,)
  return NotImplemented
 return __eq__