def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.size_alloc,self.size_free,self.succ_nodes,)==(other.size_alloc,other.size_free,other.succ_nodes,)
  return NotImplemented
 return __eq__