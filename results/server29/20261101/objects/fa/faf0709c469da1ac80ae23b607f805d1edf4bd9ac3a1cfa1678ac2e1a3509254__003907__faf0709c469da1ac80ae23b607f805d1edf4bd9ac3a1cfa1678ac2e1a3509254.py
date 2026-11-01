def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.user,self.total,self.datasets,self.models,self.spaces,)==(other.user,other.total,other.datasets,other.models,other.spaces,)
  return NotImplemented
 return __eq__