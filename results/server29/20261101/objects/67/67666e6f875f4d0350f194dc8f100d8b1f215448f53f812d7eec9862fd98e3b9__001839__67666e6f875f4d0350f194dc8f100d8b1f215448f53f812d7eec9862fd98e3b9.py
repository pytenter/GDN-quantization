def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.all_gather_metadata,self.all_gather_input_sizes,)==(other.all_gather_metadata,other.all_gather_input_sizes,)
  return NotImplemented
 return __eq__