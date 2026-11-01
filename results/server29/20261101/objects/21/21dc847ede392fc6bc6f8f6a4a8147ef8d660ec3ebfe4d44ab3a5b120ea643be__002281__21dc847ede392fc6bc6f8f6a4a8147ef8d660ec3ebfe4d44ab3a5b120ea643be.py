def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self._fx_graph_cache_key,self._time_taken_ns,self.filename,)==(other._fx_graph_cache_key,other._time_taken_ns,other.filename,)
  return NotImplemented
 return __eq__