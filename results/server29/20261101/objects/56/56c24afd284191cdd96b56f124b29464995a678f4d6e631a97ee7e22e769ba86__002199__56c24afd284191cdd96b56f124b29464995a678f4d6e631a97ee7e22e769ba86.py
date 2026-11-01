def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.num_cache_entries,self.num_cache_entries_with_same_id_matched_objs,)==(other.num_cache_entries,other.num_cache_entries_with_same_id_matched_objs,)
  return NotImplemented
 return __eq__