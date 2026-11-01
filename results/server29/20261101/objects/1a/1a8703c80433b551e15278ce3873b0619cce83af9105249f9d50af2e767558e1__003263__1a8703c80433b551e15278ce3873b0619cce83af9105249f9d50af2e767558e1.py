def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.input_tensors,self.output_tensor,)==(other.input_tensors,other.output_tensor,)
  return NotImplemented
 return __eq__