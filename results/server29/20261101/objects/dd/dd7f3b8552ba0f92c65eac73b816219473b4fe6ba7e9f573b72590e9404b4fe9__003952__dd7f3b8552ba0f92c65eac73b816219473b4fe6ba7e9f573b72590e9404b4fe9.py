def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.fullgraph,self.dynamic,self.backend,self.mode,self.options,)==(other.fullgraph,other.dynamic,other.backend,other.mode,other.options,)
  return NotImplemented
 return __eq__