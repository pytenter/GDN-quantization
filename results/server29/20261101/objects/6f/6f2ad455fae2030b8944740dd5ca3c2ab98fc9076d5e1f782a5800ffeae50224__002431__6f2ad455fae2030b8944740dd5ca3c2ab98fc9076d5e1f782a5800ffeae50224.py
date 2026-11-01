def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.input_path,self.output_path,)==(other.input_path,other.output_path,)
  return NotImplemented
 return __eq__