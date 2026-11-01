def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.name,self.index,self.var_names,self.size,self.mode,)==(other.name,other.index,other.var_names,other.size,other.mode,)
  return NotImplemented
 return __eq__