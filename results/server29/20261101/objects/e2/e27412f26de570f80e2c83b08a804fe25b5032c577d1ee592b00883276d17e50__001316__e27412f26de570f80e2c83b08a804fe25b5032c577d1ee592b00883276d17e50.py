def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.stack_index,self.target_values,)==(other.stack_index,other.target_values,)
  return NotImplemented
 return __eq__