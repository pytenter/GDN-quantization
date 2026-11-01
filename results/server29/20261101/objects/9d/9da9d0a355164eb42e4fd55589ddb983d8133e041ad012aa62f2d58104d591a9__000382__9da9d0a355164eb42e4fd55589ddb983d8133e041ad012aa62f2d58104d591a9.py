def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.by_id,self.by_key,self.by_version,self._by_id_keyset,)==(other.by_id,other.by_key,other.by_version,other._by_id_keyset,)
  return NotImplemented
 return __eq__