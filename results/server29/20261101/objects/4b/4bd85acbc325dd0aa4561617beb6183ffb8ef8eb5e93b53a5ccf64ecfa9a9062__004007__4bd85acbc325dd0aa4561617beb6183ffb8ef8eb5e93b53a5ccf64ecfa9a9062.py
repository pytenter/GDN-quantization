def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.backend,self.mode,self.fullgraph,self.dynamic,self.options,self.disable,self.use_regional_compilation,)==(other.backend,other.mode,other.fullgraph,other.dynamic,other.options,other.disable,other.use_regional_compilation,)
  return NotImplemented
 return __eq__