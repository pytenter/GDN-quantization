def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.local_name,self.is_input,self.dynamism,self.is_derefed_cell_contents,)==(other.local_name,other.is_input,other.dynamism,other.is_derefed_cell_contents,)
  return NotImplemented
 return __eq__