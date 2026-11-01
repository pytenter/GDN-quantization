def __create_fn__():
 def __lt__(self,other):
  if other.__class__ is self.__class__:
   return (self.mst_weight,)<(other.mst_weight,)
  return NotImplemented
 return __lt__