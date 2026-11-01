def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.src,self.globals,self._lineno_map,)==(other.src,other.globals,other._lineno_map,)
  return NotImplemented
 return __eq__