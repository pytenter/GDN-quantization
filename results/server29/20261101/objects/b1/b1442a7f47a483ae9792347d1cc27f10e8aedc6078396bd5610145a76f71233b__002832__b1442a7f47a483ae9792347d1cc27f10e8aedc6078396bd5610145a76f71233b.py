def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.index,self.var_names,self.size,)==(other.index,other.var_names,other.size,)
  return NotImplemented
 return __eq__