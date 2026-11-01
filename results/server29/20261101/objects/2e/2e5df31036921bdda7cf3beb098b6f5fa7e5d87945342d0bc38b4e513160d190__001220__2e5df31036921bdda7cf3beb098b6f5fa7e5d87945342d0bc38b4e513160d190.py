def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.warn_only,self.source_pairs,self.derived_equalities,self.phantom_symbols,self.relaxed_sources,self._parents,self._defs,)==(other.warn_only,other.source_pairs,other.derived_equalities,other.phantom_symbols,other.relaxed_sources,other._parents,other._defs,)
  return NotImplemented
 return __eq__