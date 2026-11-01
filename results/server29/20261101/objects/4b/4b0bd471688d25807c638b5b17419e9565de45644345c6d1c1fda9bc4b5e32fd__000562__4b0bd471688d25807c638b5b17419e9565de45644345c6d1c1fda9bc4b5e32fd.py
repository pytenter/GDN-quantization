def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.output_infos,self.is_output_tuple,)==(other.output_infos,other.is_output_tuple,)
  return NotImplemented
 return __eq__