def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.mst_weight,)==(other.mst_weight,)
  return NotImplemented
 return __eq__