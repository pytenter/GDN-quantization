def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.path,self.parts,self.parametrization,self.module_name,self.original_index,)==(other.path,other.parts,other.parametrization,other.module_name,other.original_index,)
  return NotImplemented
 return __eq__