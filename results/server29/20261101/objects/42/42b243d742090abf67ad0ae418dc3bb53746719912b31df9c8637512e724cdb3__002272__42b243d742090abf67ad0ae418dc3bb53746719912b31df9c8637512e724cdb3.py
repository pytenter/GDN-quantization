def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.configs_hash,self.local_cache,self.remote_cache,)==(other.configs_hash,other.local_cache,other.remote_cache,)
  return NotImplemented
 return __eq__